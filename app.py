import os
import sys
import io
import base64
import json
import tempfile
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP
import spacy
from textblob import TextBlob
from collections import Counter

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from pipeline import run_full_pipeline
from clustering.bertopic_clustering import cluster_with_bertopic
from analysis.llm_cluster_analysis import analyze_clusters
from clustering.semantic_clustering import generate_embeddings

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['DATA_FOLDER'] = 'data'  # Directory to save generated files

# Import werkzeug for error handling
import werkzeug

# Ensure data directory exists
os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)

# Global variables to store pipeline data
PIPELINE_CACHE = {}
EMBEDDING_MODEL = None
EMBEDDINGS_CACHE = {}

# Initialize NLP models
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy model 'en_core_web_sm' not found. NER will be limited.")
    nlp = None


def get_embedding_model():
    """Lazy load embedding model"""
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is None:
        print("Loading embedding model...")
        EMBEDDING_MODEL = SentenceTransformer('all-mpnet-base-v2')
    return EMBEDDING_MODEL


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def create_topic_count_chart(loose_data, strict_data):
    """Create chart comparing topic counts between loose and strict"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    modes = ['Loose', 'Strict']
    topic_counts = []
    
    if loose_data.get('df_clustered') is not None and 'topic' in loose_data['df_clustered'].columns:
        topic_counts.append(len([t for t in loose_data['df_clustered']['topic'].unique() if t != -1]))
    else:
        topic_counts.append(0)
    
    if strict_data.get('df_clustered') is not None and 'topic' in strict_data['df_clustered'].columns:
        topic_counts.append(len([t for t in strict_data['df_clustered']['topic'].unique() if t != -1]))
    else:
        topic_counts.append(0)
    
    if max(topic_counts) == 0:
        ax.text(0.5, 0.5, 'No topics found', ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Topics Found: Loose vs Strict Filtering', fontsize=14, fontweight='bold')
    else:
        bars = ax.bar(modes, topic_counts, color=['#22c55e', '#3b82f6'], alpha=0.8)
        ax.set_ylabel('Number of Topics', fontsize=12)
        ax.set_title('Topics Found: Loose vs Strict Filtering', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(topic_counts) * 1.2 if max(topic_counts) > 0 else 1)
        
        # Add value labels on bars
        for bar, count in zip(bars, topic_counts):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{count}',
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig_to_base64(fig)


def create_coherence_comparison_chart(loose_data, strict_data):
    """Create chart comparing coherence scores between loose and strict"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Coherence Scores Comparison: Loose vs Strict', fontsize=16, fontweight='bold')
    
    coherence_metrics = [
        'llm_overall_coherence_score',
        'llm_semantic_coherence',
        'llm_topical_focus',
        'llm_lexical_cohesion',
        'llm_lexical_informativeness',
        'llm_outlier_presence'
    ]
    
    metric_labels = [
        'Overall Coherence',
        'Semantic Coherence',
        'Topical Focus',
        'Lexical Cohesion',
        'Lexical Informativeness',
        'Outlier Presence'
    ]
    
    for idx, (metric, label) in enumerate(zip(coherence_metrics, metric_labels)):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        loose_scores = []
        strict_scores = []
        
        if (loose_data.get('df_analyzed') is not None and 
            metric in loose_data['df_analyzed'].columns and 
            'topic' in loose_data['df_analyzed'].columns):
            topic_filter = loose_data['df_analyzed']['topic'] != -1
            loose_scores = loose_data['df_analyzed'][topic_filter][metric].dropna().tolist()
        
        if (strict_data.get('df_analyzed') is not None and 
            metric in strict_data['df_analyzed'].columns and 
            'topic' in strict_data['df_analyzed'].columns):
            topic_filter = strict_data['df_analyzed']['topic'] != -1
            strict_scores = strict_data['df_analyzed'][topic_filter][metric].dropna().tolist()
        
        if loose_scores or strict_scores:
            data_to_plot = [loose_scores, strict_scores]
            bp = ax.boxplot(data_to_plot, labels=['Loose', 'Strict'], patch_artist=True)
            bp['boxes'][0].set_facecolor('#22c55e')
            bp['boxes'][1].set_facecolor('#3b82f6')
            bp['boxes'][0].set_alpha(0.7)
            bp['boxes'][1].set_alpha(0.7)
            
            ax.set_ylabel('Score', fontsize=10)
            ax.set_title(label, fontsize=11, fontweight='bold')
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label, fontsize=11)
    
    plt.tight_layout()
    return fig_to_base64(fig)


def create_topic_size_distribution(loose_data, strict_data):
    """Create chart showing topic size distribution"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Topic Size Distribution: Loose vs Strict', fontsize=14, fontweight='bold')
    
    for idx, (data, mode, ax, color) in enumerate([
        (loose_data, 'Loose', ax1, '#22c55e'),
        (strict_data, 'Strict', ax2, '#3b82f6')
    ]):
        if (data.get('df_clustered') is not None and 
            'topic' in data['df_clustered'].columns):
            topic_counts = data['df_clustered'][data['df_clustered']['topic'] != -1]['topic'].value_counts().values
            
            if len(topic_counts) > 0:
                ax.hist(topic_counts, bins=min(20, len(topic_counts)), color=color, alpha=0.7, edgecolor='black')
                ax.set_xlabel('Documents per Topic', fontsize=11)
                ax.set_ylabel('Number of Topics', fontsize=11)
                ax.set_title(f'{mode} Filtering', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No topics found', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    return fig_to_base64(fig)


def create_coherence_scatter(loose_data, strict_data):
    """Create scatter plot comparing overall coherence vs topic size"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'loose': '#22c55e', 'strict': '#3b82f6'}
    
    for mode, data, label in [
        ('loose', loose_data, 'Loose'),
        ('strict', strict_data, 'Strict')
    ]:
        if (data.get('df_analyzed') is not None and 
            data.get('df_clustered') is not None and
            'topic' in data['df_clustered'].columns):
            df_analyzed = data['df_analyzed']
            df_clustered = data['df_clustered']
            
            # Get topic sizes
            topic_sizes = df_clustered[df_clustered['topic'] != -1]['topic'].value_counts()
            
            # Get coherence scores per topic
            if 'llm_overall_coherence_score' in df_analyzed.columns:
                topic_coherence = df_analyzed[df_analyzed['topic'] != -1].groupby('topic')['llm_overall_coherence_score'].mean()
            else:
                # No coherence scores available, skip this mode
                continue
            
            # Match sizes with coherence
            sizes = []
            coherences = []
            for topic_id in topic_coherence.index:
                if topic_id in topic_sizes.index:
                    sizes.append(topic_sizes[topic_id])
                    coherences.append(topic_coherence[topic_id])
            
            if sizes and coherences:
                ax.scatter(sizes, coherences, label=label, color=colors[mode], alpha=0.6, s=100)
    
    ax.set_xlabel('Topic Size (Number of Documents)', fontsize=12)
    ax.set_ylabel('Overall Coherence Score', fontsize=12)
    ax.set_title('Topic Size vs Coherence: Loose vs Strict', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig_to_base64(fig)


def analyze_sentiment(df, column_name='content_text'):
    """Analyze sentiment of documents"""
    texts = df[column_name].astype(str).tolist()
    
    sentiments = []
    for text in texts[:1000]:  # Limit for performance
        blob = TextBlob(text)
        sentiments.append({
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        })
    
    if not sentiments:
        return {
            'overall_polarity': 0.0,
            'overall_subjectivity': 0.0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0
        }
    
    avg_polarity = np.mean([s['polarity'] for s in sentiments])
    avg_subjectivity = np.mean([s['subjectivity'] for s in sentiments])
    
    positive = sum(1 for s in sentiments if s['polarity'] > 0.1)
    negative = sum(1 for s in sentiments if s['polarity'] < -0.1)
    neutral = len(sentiments) - positive - negative
    
    return {
        'overall_polarity': float(avg_polarity),
        'overall_subjectivity': float(avg_subjectivity),
        'positive_count': positive,
        'negative_count': negative,
        'neutral_count': neutral,
        'total_analyzed': len(sentiments)
    }


def analyze_entities(df, column_name='content_text'):
    """Extract named entities from documents"""
    if nlp is None:
        return {
            'total_entities': 0,
            'entity_types': {},
            'top_entities': []
        }
    
    texts = df[column_name].astype(str).tolist()[:500]  # Limit for performance
    
    all_entities = []
    entity_types = Counter()
    
    for text in texts:
        doc = nlp(text)
        for ent in doc.ents:
            all_entities.append(ent.text)
            entity_types[ent.label_] += 1
    
    top_entities = Counter(all_entities).most_common(20)
    
    return {
        'total_entities': len(all_entities),
        'entity_types': dict(entity_types),
        'top_entities': [{'text': ent, 'count': count} for ent, count in top_entities]
    }


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors"""
    max_size_mb = app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
    return jsonify({
        'error': f'File too large. Maximum size is {max_size_mb:.0f}MB. Please use a smaller file or compress your data.'
    }), 413


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze/', methods=['POST'])
def analyze():
    """Main endpoint to analyze uploaded CSV file"""
    try:
        # Handle file size errors
        if request.content_length and request.content_length > app.config['MAX_CONTENT_LENGTH']:
            return jsonify({
                'error': f'File too large. Maximum size is {app.config["MAX_CONTENT_LENGTH"] / (1024*1024):.0f}MB'
            }), 413
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        # Convert CSV to Excel if needed, or use xlsx directly
        if filename.endswith('.csv'):
            df = pd.read_csv(input_path)
            excel_path = input_path.replace('.csv', '.xlsx')
            df.to_excel(excel_path, index=False)
            input_path = excel_path
        elif filename.endswith('.xlsx'):
            # Already xlsx, use directly
            pass
        else:
            return jsonify({'error': 'Unsupported file format. Please upload CSV or XLSX.'}), 400
        
        # Read file to detect columns
        df_test = pd.read_excel(input_path, nrows=1)
        
        # Determine text column
        possible_columns = ['content_text', 'text', 'content', 'Text', 'Content']
        column_name = 'content_text'
        for col in possible_columns:
            if col in df_test.columns:
                column_name = col
                break
        
        # Detect label and country columns
        label_column = None
        country_column = None
        possible_labels = ['Label', 'label', 'labels', 'Labels', 'category', 'Category']
        possible_countries = ['Country', 'country', 'countries', 'Countries', 'location', 'Location']
        
        for col in possible_labels:
            if col in df_test.columns:
                label_column = col
                break
        
        for col in possible_countries:
            if col in df_test.columns:
                country_column = col
                break
        
        # Run pipeline (it runs both loose and strict internally)
        print("Running full pipeline (loose and strict modes)...")
        print(f"Detected columns - Text: {column_name}, Label: {label_column}, Country: {country_column}")
        
        # Generate unique timestamp for this run to avoid overwriting
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = os.path.splitext(filename)[0]
        run_prefix = f"{base_name}_{timestamp}"
        
        print(f"\n{'='*60}")
        print(f"Saving output files to: {app.config['DATA_FOLDER']}/")
        print(f"File prefix: {run_prefix}")
        print(f"{'='*60}\n")
        
        pipeline_kwargs = {
            'preprocess_input_path': input_path,
            'preprocess_output_path': os.path.join(app.config['DATA_FOLDER'], f'{run_prefix}_cleaned_data.xlsx'),
            'bertopic_output_path': os.path.join(app.config['DATA_FOLDER'], f'{run_prefix}_clustered_data.xlsx'),
            'llm_output_path': os.path.join(app.config['DATA_FOLDER'], f'{run_prefix}_analyzed_data.xlsx'),
            'column_name': column_name,
            'bertopic_min_topic_size': 5,
            'llm_api_key': os.getenv('OPENAI_API_KEY')  # Pass API key if available
        }
        
        # Add label and country parameters if detected
        if label_column:
            pipeline_kwargs['preprocess_label_column'] = label_column
        if country_column:
            pipeline_kwargs['preprocess_country_column'] = country_column
        
        results = run_full_pipeline(**pipeline_kwargs)
        
        # Store results - pipeline returns dict with 'loose' and 'strict' keys
        loose_tuple = results.get('loose', (None, None, None))
        strict_tuple = results.get('strict', (None, None, None))
        
        loose_data = {
            'df_cleaned': loose_tuple[0] if len(loose_tuple) > 0 else None,
            'df_clustered': loose_tuple[1] if len(loose_tuple) > 1 else None,
            'df_analyzed': loose_tuple[2] if len(loose_tuple) > 2 else loose_tuple[1] if len(loose_tuple) > 1 else None
        }
        
        strict_data = {
            'df_cleaned': strict_tuple[0] if len(strict_tuple) > 0 else None,
            'df_clustered': strict_tuple[1] if len(strict_tuple) > 1 else None,
            'df_analyzed': strict_tuple[2] if len(strict_tuple) > 2 else strict_tuple[1] if len(strict_tuple) > 1 else None
        }
        
        # Generate visualizations
        charts = []
        
        # Main chart: Topic count comparison
        chart1 = create_topic_count_chart(loose_data, strict_data)
        charts.append({
            'title': 'Topics Found: Loose vs Strict',
            'imageBase64': chart1
        })
        
        # Coherence comparison
        chart2 = create_coherence_comparison_chart(loose_data, strict_data)
        charts.append({
            'title': 'Coherence Scores Comparison',
            'imageBase64': chart2
        })
        
        # Topic size distribution
        chart3 = create_topic_size_distribution(loose_data, strict_data)
        charts.append({
            'title': 'Topic Size Distribution',
            'imageBase64': chart3
        })
        
        # Coherence vs size scatter
        chart4 = create_coherence_scatter(loose_data, strict_data)
        charts.append({
            'title': 'Topic Size vs Coherence',
            'imageBase64': chart4
        })
        
        # Dataset insights
        dfs_to_combine = []
        if loose_data['df_cleaned'] is not None:
            dfs_to_combine.append(loose_data['df_cleaned'])
        if strict_data['df_cleaned'] is not None:
            dfs_to_combine.append(strict_data['df_cleaned'])
        
        if dfs_to_combine:
            combined_df = pd.concat(dfs_to_combine).drop_duplicates()
        else:
            # Fallback: read original file
            combined_df = pd.read_excel(input_path)
        
        text_lengths = combined_df[column_name].astype(str).str.len()
        
        dataset_insights = {
            'num_documents': len(combined_df),
            'avg_length': int(text_lengths.mean()),
            'median_length': int(text_lengths.median())
        }
        
        # Sentiment analysis (using combined data)
        sentiment_overall = analyze_sentiment(combined_df, column_name)
        
        # Entity analysis
        entity_overall = analyze_entities(combined_df, column_name)
        
        # Topic cards (from strict mode as it's more refined, fallback to loose)
        topic_cards = {}
        for data in [strict_data, loose_data]:
            if (data.get('df_analyzed') is not None and 
                'llm_topic_label' in data['df_analyzed'].columns and
                'topic' in data['df_analyzed'].columns):
                for topic_id in data['df_analyzed']['topic'].unique():
                    if topic_id != -1:
                        topic_df = data['df_analyzed'][data['df_analyzed']['topic'] == topic_id]
                        if len(topic_df) > 0:
                            label = topic_df['llm_topic_label'].iloc[0]
                            if label and label != 'N/A':
                                summary = topic_df['llm_summary'].iloc[0] if 'llm_summary' in topic_df.columns else f"Topic {topic_id}"
                                topic_cards[label] = {'summary': summary}
                if topic_cards:  # If we found topics, break
                    break
        
        # Store for search and PDF
        cache_key = filename
        PIPELINE_CACHE[cache_key] = {
            'loose_data': loose_data,
            'strict_data': strict_data,
            'column_name': column_name
        }
        
        # Generate embeddings for search - DISABLED (causes hang)
        # print("Generating embeddings for semantic search...")
        # model = get_embedding_model()
        # texts = combined_df[column_name].astype(str).tolist()
        # embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
        # EMBEDDINGS_CACHE[cache_key] = {
        #     'embeddings': embeddings,
        #     'texts': texts,
        #     'df': combined_df
        # }
        print("Embedding generation disabled - semantic search will not be available")
        
        response = {
            'charts': charts,
            'dataset_insights': dataset_insights,
            'summary': {
                'sentiment_overall': sentiment_overall,
                'entity_overall': entity_overall
            },
            'topic_cards': topic_cards
        }
        
        # Store cache_key separately for search (not needed in frontend response)
        # but we'll include it for potential future use
        if cache_key:
            response['cache_key'] = cache_key
        
        return jsonify(response)
        
    except werkzeug.exceptions.RequestEntityTooLarge:
        max_size_mb = app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
        return jsonify({
            'error': f'File too large. Maximum size is {max_size_mb:.0f}MB. Please use a smaller file or compress your data.'
        }), 413
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = str(e)
        if 'Request Entity Too Large' in error_msg or '413' in error_msg:
            max_size_mb = app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
            return jsonify({
                'error': f'File too large. Maximum size is {max_size_mb:.0f}MB. Please use a smaller file or compress your data.'
            }), 413
        return jsonify({'error': error_msg}), 500


@app.route('/api/search/', methods=['POST'])
def search():
    """Semantic search endpoint - DISABLED (embedding generation causes hang)"""
    return jsonify({
        'error': 'Semantic search is currently disabled. Embedding generation has been disabled to prevent hangs.',
        'results': []
    }), 503


@app.route('/api/report/', methods=['GET'])
def report():
    """Generate PDF report"""
    # For now, return a placeholder
    # In a full implementation, you would use reportlab or similar
    return jsonify({'message': 'PDF report generation not yet implemented'})


if __name__ == '__main__':
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    app.run(debug=True, port=5000)

