import pandas as pd
import os
from collections import Counter
import re
from openai import OpenAI


def get_top_ngrams(texts, n=1, top_k=10):
    """
    Extract top unigrams or bigrams from a list of texts.
    
    Parameters:
    -----------
    texts : list
        List of text strings
    n : int, default=1
        1 for unigrams, 2 for bigrams
    top_k : int, default=10
        Number of top ngrams to return
        
    Returns:
    --------
    list
        List of top ngrams with their counts
    """
    all_ngrams = []
    
    for text in texts:
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        if n == 1:
            ngrams = words
        else:
            ngrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        all_ngrams.extend(ngrams)
    
    ngram_counts = Counter(all_ngrams)
    return ngram_counts.most_common(top_k)


def analyze_cluster_with_llm(representative_texts, unigrams, bigrams, metadata_info, 
                            client, model="gpt-4o-mini"):
    """
    Analyze a cluster using LLM with a one-shot prompt.
    
    Parameters:
    -----------
    representative_texts : list
        List of 5 representative texts from the cluster
    unigrams : list
        Top unigrams with counts
    bigrams : list
        Top bigrams with counts
    metadata_info : dict
        Dictionary with metadata (labels, country, domain counts)
    client : OpenAI client
        OpenAI client instance
    model : str, default="gpt-4o-mini"
        Model to use for analysis
        
    Returns:
    --------
    dict
        Dictionary with topic_label, summary, coherence, and merge_suggestions
    """
    
    unigrams_str = ', '.join([f"{word} ({count})" for word, count in unigrams[:10]])
    bigrams_str = ', '.join([f"{bigram} ({count})" for bigram, count in bigrams[:10]])
    
    metadata_str = f"Labels: {metadata_info.get('labels', 'N/A')}, "
    metadata_str += f"Countries: {metadata_info.get('countries', 'N/A')}, "
    metadata_str += f"Domains: {metadata_info.get('domains', 'N/A')}"
    
    texts_str = "\n\n".join([f"Text {i+1}:\n{text[:500]}" for i, text in enumerate(representative_texts[:5])])
    
    prompt = f"""Analyze the following cluster of documents and provide a structured analysis.

Cluster Information:
- Top Unigrams: {unigrams_str}
- Top Bigrams: {bigrams_str}
- Metadata: {metadata_str}

Representative Texts:
{texts_str}

Please provide:
1. Topic Label: A concise, descriptive label for this cluster (e.g., "Ethical concerns & misuse", "AI in student services")
2. Summary: A 2-3 sentence summary of what this cluster is about
3. Coherence Assessment: Is this cluster coherent? (Yes/No with brief explanation)
4. Merge/Split Suggestions: Should this cluster be merged with others or split? Provide specific suggestions.

Format your response as:
TOPIC_LABEL: [label]
SUMMARY: [summary]
COHERENCE: [assessment]
MERGE_SPLIT: [suggestions]"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert at analyzing document clusters and identifying themes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        result_text = response.choices[0].message.content
        
        result = {
            'topic_label': '',
            'summary': '',
            'coherence': '',
            'merge_suggestions': ''
        }
        
        for line in result_text.split('\n'):
            if line.startswith('TOPIC_LABEL:'):
                result['topic_label'] = line.replace('TOPIC_LABEL:', '').strip()
            elif line.startswith('SUMMARY:'):
                result['summary'] = line.replace('SUMMARY:', '').strip()
            elif line.startswith('COHERENCE:'):
                result['coherence'] = line.replace('COHERENCE:', '').strip()
            elif line.startswith('MERGE_SPLIT:'):
                result['merge_suggestions'] = line.replace('MERGE_SPLIT:', '').strip()
        
        return result
        
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return {
            'topic_label': 'Error in analysis',
            'summary': f'Error: {str(e)}',
            'coherence': 'Unknown',
            'merge_suggestions': 'N/A'
        }


def analyze_clusters(data_path='data/bertopic_clustered_data.xlsx',
                    column_name='content_text',
                    output_path='data/llm_analyzed_clusters.xlsx',
                    api_key=None,
                    model="gpt-4o-mini"):
    """
    Analyze clusters using LLM to get topic labels, summaries, and merge suggestions.
    
    Parameters:
    -----------
    data_path : str, default='data/bertopic_clustered_data.xlsx'
        Path to clustered data from BERTopic
    column_name : str, default='content_text'
        Name of the column containing text content
    output_path : str, default='data/llm_analyzed_clusters.xlsx'
        Path to save analyzed data
    api_key : str, optional
        OpenAI API key. If None, uses OPENAI_API_KEY environment variable
    model : str, default="gpt-4o-mini"
        OpenAI model to use
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with LLM analysis added
    """
    print("\n" + "="*60)
    print("LLM Cluster Analysis")
    print("="*60)
    
    print(f"\nLoading clustered data from {data_path}...")
    df = pd.read_excel(data_path)
    print(f"Loaded {len(df)} rows")
    
    if 'topic' not in df.columns:
        raise ValueError("DataFrame must have a 'topic' column from BERTopic clustering")
    
    if api_key is None:
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key is None:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key parameter")
    
    client = OpenAI(api_key=api_key)
    
    unique_topics = sorted([t for t in df['topic'].unique() if t != -1])
    n_topics = len(unique_topics)
    
    print(f"\nAnalyzing {n_topics} clusters...")
    print("This may take a while depending on the number of clusters...")
    
    analysis_results = {}
    
    for idx, topic_id in enumerate(unique_topics, 1):
        print(f"\n[{idx}/{n_topics}] Analyzing Topic {topic_id}...")
        
        topic_df = df[df['topic'] == topic_id].copy()
        
        if len(topic_df) == 0:
            continue
        
        texts = topic_df[column_name].astype(str).tolist()
        
        representative_texts = texts[:5]
        if len(texts) < 5:
            representative_texts = texts
        
        unigrams = get_top_ngrams(texts, n=1, top_k=10)
        bigrams = get_top_ngrams(texts, n=2, top_k=10)
        
        metadata_info = {
            'labels': ', '.join(topic_df['Label'].value_counts().head(3).index.astype(str).tolist()) if 'Label' in topic_df.columns else 'N/A',
            'countries': ', '.join(topic_df['Country'].value_counts().head(3).index.astype(str).tolist()) if 'Country' in topic_df.columns else 'N/A',
            'domains': ', '.join(topic_df['Domain'].value_counts().head(3).index.astype(str).tolist()) if 'Domain' in topic_df.columns else 'N/A'
        }
        
        result = analyze_cluster_with_llm(
            representative_texts=representative_texts,
            unigrams=unigrams,
            bigrams=bigrams,
            metadata_info=metadata_info,
            client=client,
            model=model
        )
        
        analysis_results[topic_id] = result
        
        print(f"  Topic Label: {result['topic_label']}")
        print(f"  Coherence: {result['coherence'][:100]}...")
    
    print("\n" + "="*60)
    print("Adding analysis results to DataFrame...")
    
    df['llm_topic_label'] = df['topic'].map(lambda x: analysis_results.get(x, {}).get('topic_label', 'N/A') if x != -1 else 'Noise/Outliers')
    df['llm_summary'] = df['topic'].map(lambda x: analysis_results.get(x, {}).get('summary', 'N/A') if x != -1 else 'Noise/Outliers')
    df['llm_coherence'] = df['topic'].map(lambda x: analysis_results.get(x, {}).get('coherence', 'N/A') if x != -1 else 'Noise/Outliers')
    df['llm_merge_suggestions'] = df['topic'].map(lambda x: analysis_results.get(x, {}).get('merge_suggestions', 'N/A') if x != -1 else 'N/A')
    
    print("\nSaving analyzed data...")
    df.to_excel(output_path, index=False)
    print(f"Analyzed data saved to {output_path}")
    
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    for topic_id in unique_topics:
        if topic_id in analysis_results:
            result = analysis_results[topic_id]
            print(f"\nTopic {topic_id}:")
            print(f"  Label: {result['topic_label']}")
            print(f"  Summary: {result['summary'][:150]}...")
            print(f"  Coherence: {result['coherence'][:100]}...")
            if result['merge_suggestions'] and result['merge_suggestions'] != 'N/A':
                print(f"  Merge/Split: {result['merge_suggestions'][:100]}...")
    print("\n" + "="*60)
    
    return df


def main(data_path='data/bertopic_clustered_data.xlsx',
         output_path='data/llm_analyzed_clusters.xlsx',
         api_key=None,
         model="gpt-4o-mini",
         column_name='content_text'):
    """
    Main function to run LLM cluster analysis.
    
    Parameters:
    -----------
    data_path : str, default='data/bertopic_clustered_data.xlsx'
        Path to clustered data from BERTopic
    output_path : str, default='data/llm_analyzed_clusters.xlsx'
        Path to save analyzed data
    api_key : str, optional
        OpenAI API key. If None, uses OPENAI_API_KEY environment variable
    model : str, default="gpt-4o-mini"
        OpenAI model to use
    column_name : str, default='content_text'
        Name of the text column
    """
    print("\n" + "="*80)
    print("LLM CLUSTER ANALYSIS")
    print("="*80)
    
    df_analyzed = analyze_clusters(
        data_path=data_path,
        column_name=column_name,
        output_path=output_path,
        api_key=api_key,
        model=model
    )
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nOutput file: {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()

