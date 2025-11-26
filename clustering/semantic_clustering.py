import pandas as pd
import numpy as np
import os
import sys
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from cleaning.preprocess_data import preprocess_data


def generate_embeddings(df=None, data_path='data/cleaned_data.xlsx', column_name='content_text', 
                        model_name='all-mpnet-base-v2', batch_size=32, output_path='data/embeddings.npy'):
    """
    Generate semantic embeddings for text data using the specified model.
    
    Parameters:
    -----------
    df : pandas.DataFrame, optional
        DataFrame with cleaned data. If None, loads from data_path
    data_path : str, default='data/cleaned_data.xlsx'
        Path to the cleaned data Excel file (used only if df is None)
    column_name : str, default='content_text'
        Name of the column containing text content
    model_name : str, default='all-mpnet-base-v2'
        Name of the sentence transformer model to use (e.g., 'all-mpnet-base-v2', 'all-MiniLM-L6-v2', 'intfloat/e5-large')
    batch_size : int, default=32
        Batch size for embedding generation
    output_path : str, default='data/embeddings.npy'
        Path to save the embeddings numpy array
        
    Returns:
    --------
    tuple: (pandas.DataFrame, numpy.ndarray)
        DataFrame with the data and the embeddings array
    """
    print("\n" + "="*60)
    print("Generating Semantic Embeddings...")
    print("="*60)
    
    if df is None:
        print(f"\nLoading data from {data_path}...")
        df = pd.read_excel(data_path)
        print(f"Loaded {len(df)} rows")
    else:
        print(f"\nUsing provided DataFrame with {len(df)} rows")
    
    if column_name not in df.columns:
        print(f"[ERROR] Column '{column_name}' not found. Available columns: {df.columns.tolist()}")
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df = df.dropna(subset=[column_name])
    df = df[df[column_name].astype(str).str.strip() != '']
    print(f"Rows with valid text: {len(df)}")
    
    texts = df[column_name].astype(str).tolist()
    
    e5_models = ['e5-', 'intfloat/e5']
    needs_prefix = any(e5 in model_name.lower() for e5 in e5_models)
    
    if needs_prefix:
        print(f"\nDetected E5 model - adding 'passage: ' prefix to texts...")
        texts = [f"passage: {text}" for text in texts]
    else:
        print(f"\nUsing standard sentence transformer model (no prefix needed)...")
    
    print(f"\nLoading model: {model_name}...")
    print("This may take a while on first run (downloading model)...")
    model = SentenceTransformer(model_name)
    
    print(f"\nGenerating embeddings (batch size: {batch_size})...")
    print("This may take several minutes depending on data size...")
    embeddings = model.encode(texts, batch_size=batch_size, 
                             show_progress_bar=True, convert_to_numpy=True)
    
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    print(f"\nSaving embeddings to {output_path}...")
    np.save(output_path, embeddings)
    print("Embeddings saved successfully!")
    
    print("\n" + "="*60)
    print("Embedding generation complete!")
    print("="*60)
    
    return df, embeddings


def cluster_and_extract_topics(df, embeddings, min_cluster_size=10, min_samples=5,
                              cluster_selection_epsilon=0.0, metric='euclidean',
                              column_name='content_text', output_path='data/clustered_data.xlsx'):
    """
    Apply HDBSCAN clustering to embeddings and extract topics for each cluster.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with the text data
    embeddings : numpy.ndarray
        Embeddings array from generate_embeddings
    min_cluster_size : int, default=10
        Minimum cluster size for HDBSCAN
    min_samples : int, default=5
        Minimum samples parameter for HDBSCAN
    cluster_selection_epsilon : float, default=0.0
        Epsilon parameter for cluster selection
    metric : str, default='euclidean'
        Distance metric for HDBSCAN ('euclidean', 'manhattan', 'cosine', etc.)
    column_name : str, default='content_text'
        Name of the column containing text content
    output_path : str, default='data/clustered_data.xlsx'
        Path to save the clustered data with cluster labels
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with cluster labels and topic information
    """
    print("\n" + "="*60)
    print("Applying HDBSCAN Clustering...")
    print("="*60)
    
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Number of documents: {len(df)}")
    
    print(f"\nClustering parameters:")
    print(f"  min_cluster_size: {min_cluster_size}")
    print(f"  min_samples: {min_samples}")
    print(f"  metric: {metric}")
    print(f"  cluster_selection_epsilon: {cluster_selection_epsilon}")
    
    print("\nRunning HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric=metric,
        prediction_data=True
    )
    
    cluster_labels = clusterer.fit_predict(embeddings)
    
    df_clustered = df.copy()
    df_clustered['cluster_label'] = cluster_labels
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f"\nClustering complete!")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Number of noise points (outliers): {n_noise}")
    print(f"  Total points: {len(cluster_labels)}")
    
    print("\n" + "="*60)
    print("Extracting Topics for Each Cluster...")
    print("="*60)
    
    cluster_topics = {}
    cluster_stats = {}
    
    unique_clusters = sorted(set(cluster_labels))
    if -1 in unique_clusters:
        unique_clusters.remove(-1)
        unique_clusters.append(-1)  # Put noise cluster at the end
    
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            print(f"\n[Noise Cluster (-1)]")
            print(f"  Number of documents: {n_noise}")
            cluster_topics[cluster_id] = "Noise/Outliers"
            cluster_stats[cluster_id] = {
                'count': n_noise,
                'topic': "Noise/Outliers",
                'representative_texts': []
            }
            continue
        
        cluster_mask = cluster_labels == cluster_id
        cluster_embeddings = embeddings[cluster_mask]
        cluster_texts = df_clustered[cluster_mask][column_name].tolist()
        
        print(f"\n[Cluster {cluster_id}]")
        print(f"  Number of documents: {len(cluster_texts)}")
        
        cluster_centroid = cluster_embeddings.mean(axis=0)
        similarities = cosine_similarity([cluster_centroid], cluster_embeddings)[0]
        most_representative_idx = np.argmax(similarities)
        representative_text = cluster_texts[most_representative_idx]
        
        all_text = ' '.join(cluster_texts).lower()
        words = re.findall(r'\b[a-z]{4,}\b', all_text)
        word_freq = Counter(words)
        common_stopwords = {'that', 'this', 'with', 'from', 'have', 'been', 'were', 
                           'their', 'there', 'these', 'would', 'could', 'should',
                           'which', 'other', 'about', 'after', 'before', 'during'}
        filtered_words = {word: count for word, count in word_freq.items() 
                         if word not in common_stopwords}
        top_terms = [word for word, _ in Counter(filtered_words).most_common(10)]
        
        topic_description = f"Topic: {', '.join(top_terms[:5])}"
        if len(representative_text) > 200:
            topic_description += f"\nRepresentative text: {representative_text[:200]}..."
        else:
            topic_description += f"\nRepresentative text: {representative_text}"
        
        cluster_topics[cluster_id] = topic_description
        cluster_stats[cluster_id] = {
            'count': len(cluster_texts),
            'topic': topic_description,
            'top_terms': top_terms[:10],
            'representative_text': representative_text[:500]
        }
        
        print(f"  Top terms: {', '.join(top_terms[:10])}")
        print(f"  Representative text preview: {representative_text[:150]}...")
    
    df_clustered['cluster_topic'] = df_clustered['cluster_label'].map(cluster_topics)
    
    print(f"\nSaving clustered data to {output_path}...")
    df_clustered.to_excel(output_path, index=False)
    print("Clustered data saved successfully!")
    
    print("\n" + "="*60)
    print("CLUSTERING SUMMARY")
    print("="*60)
    print(f"\nTotal number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise}")
    print(f"\nCluster Details:")
    print("-" * 60)
    
    for cluster_id in sorted([c for c in unique_clusters if c != -1]):
        stats = cluster_stats[cluster_id]
        print(f"\nCluster {cluster_id}:")
        print(f"  Documents: {stats['count']}")
        print(f"  Top terms: {', '.join(stats['top_terms'][:8])}")
        print(f"  Representative text: {stats['representative_text'][:200]}...")
    
    if -1 in unique_clusters:
        print(f"\nNoise Cluster (-1):")
        print(f"  Documents: {n_noise}")
        print(f"  These are outliers that don't fit into any cluster")
    
    print("\n" + "="*60)
    
    return df_clustered, cluster_stats


def main(preprocess_input_path='data/Final_table_results.xlsx',
         preprocess_output_path='data/cleaned_data.xlsx',
         preprocess_label_column='Label',
         preprocess_country_column='Country',
         preprocess_target_country='United States',
         preprocess_valid_labels=None,
         preprocess_filtering_mode='strict',
         preprocess_thresholds=None,
         embedding_model='intfloat/e5-large', # intfloat/e5-large, all-mpnet-base-v2, all-mpnet-base-v2
         embedding_batch_size=32,
         embedding_output_path='data/embeddings.npy',
         clustering_min_cluster_size=10,
         clustering_min_samples=5,
         clustering_epsilon=0.0,
         clustering_metric='euclidean',
         clustering_output_path='data/clustered_data.xlsx',
         column_name='content_text'):
    """
    Main function to run preprocessing, embedding generation, and clustering sequentially.
    
    Parameters:
    -----------
    preprocess_input_path : str, default='data/Final_table_results.xlsx'
        Path to input data for preprocessing
    preprocess_output_path : str, default='data/cleaned_data.xlsx'
        Path to save cleaned data
    preprocess_label_column : str, default='Label'
        Name of the column containing labels
    preprocess_country_column : str, default='Country'
        Name of the column containing country information
    preprocess_target_country : str, default='United States'
        The country to filter for (case-insensitive)
    preprocess_valid_labels : set or list, optional
        Set or list of valid labels to filter for. If None, defaults to 
        {'research materials', 'technical documentation'}
    preprocess_filtering_mode : str, default='loose'
        Preprocessing filtering mode: 'strict', 'loose', or 'none'
    preprocess_thresholds : dict, optional
        Custom thresholds for preprocessing
    embedding_model : str, default='all-mpnet-base-v2'
        Model name for embeddings
    embedding_batch_size : int, default=32
        Batch size for embedding generation
    embedding_output_path : str, default='data/embeddings.npy'
        Path to save embeddings
    clustering_min_cluster_size : int, default=10
        Minimum cluster size for HDBSCAN
    clustering_min_samples : int, default=5
        Minimum samples for HDBSCAN
    clustering_epsilon : float, default=0.0
        Cluster selection epsilon for HDBSCAN
    clustering_metric : str, default='euclidean'
        Distance metric for HDBSCAN
    clustering_output_path : str, default='data/clustered_data.xlsx'
        Path to save clustered data
    column_name : str, default='content_text'
        Name of the text column
    """
    print("\n" + "="*80)
    print("SEMANTIC CLUSTERING PIPELINE")
    print("="*80)
    
    print("\n[STEP 0/2] Preprocessing data...")
    df_cleaned = preprocess_data(
        input_path=preprocess_input_path,
        output_path=preprocess_output_path,
        label_column=preprocess_label_column,
        country_column=preprocess_country_column,
        target_country=preprocess_target_country,
        valid_labels=preprocess_valid_labels,
        filtering_mode=preprocess_filtering_mode,
        thresholds=preprocess_thresholds,
        column_name=column_name,
        save_output=True
    )
    
    print("\n[STEP 1/2] Generating embeddings...")
    df, embeddings = generate_embeddings(
        df=df_cleaned,
        data_path=preprocess_output_path,
        column_name=column_name,
        model_name=embedding_model,
        batch_size=embedding_batch_size,
        output_path=embedding_output_path
    )
    
    print("\n[STEP 2/2] Clustering and extracting topics...")
    df_clustered, cluster_stats = cluster_and_extract_topics(
        df=df,
        embeddings=embeddings,
        min_cluster_size=clustering_min_cluster_size,
        min_samples=clustering_min_samples,
        cluster_selection_epsilon=clustering_epsilon,
        metric=clustering_metric,
        column_name=column_name,
        output_path=clustering_output_path
    )
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - Cleaned data: {preprocess_output_path}")
    print(f"  - Embeddings: {embedding_output_path}")
    print(f"  - Clustered data: {clustering_output_path}")
    print(f"\nTotal clusters found: {len([c for c in cluster_stats.keys() if c != -1])}")
    print("="*80)


if __name__ == "__main__":
    main()

