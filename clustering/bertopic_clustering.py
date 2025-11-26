import pandas as pd
import os
import sys
from bertopic import BERTopic

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from cleaning.preprocess_data import preprocess_data


def cluster_with_bertopic(data_path='data/cleaned_data.xlsx', 
                          column_name='content_text',
                          output_path='data/bertopic_clustered_data.xlsx',
                          min_topic_size=10,
                          run_preprocessing=False,
                          preprocess_input_path='data/Final_table_results.xlsx',
                          preprocess_output_path='data/cleaned_data.xlsx',
                          preprocess_label_column='Label',
                          preprocess_country_column='Country',
                          preprocess_target_country='United States',
                          preprocess_valid_labels=None,
                          preprocess_filtering_mode='strict',
                          preprocess_thresholds=None):
    """
    Cluster text data using BERTopic.
    
    Parameters:
    -----------
    data_path : str, default='data/cleaned_data.xlsx'
        Path to the cleaned data Excel file (used if run_preprocessing=False)
    column_name : str, default='content_text'
        Name of the column containing text content
    output_path : str, default='data/bertopic_clustered_data.xlsx'
        Path to save the clustered data
    min_topic_size : int, default=10
        Minimum size of topics (similar to min_cluster_size)
    run_preprocessing : bool, default=True
        Whether to run preprocessing before clustering
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
    preprocess_filtering_mode : str, default='strict'
        Preprocessing filtering mode: 'strict', 'loose', or 'none'
    preprocess_thresholds : dict, optional
        Custom thresholds for preprocessing
        
    Returns:
    --------
    tuple: (pandas.DataFrame, BERTopic)
        DataFrame with cluster labels and topic information, and the BERTopic model
    """
    print("\n" + "="*60)
    print("BERTopic Clustering")
    print("="*60)
    
    if run_preprocessing:
        print("\n[STEP 0/1] Running preprocessing...")
        df = preprocess_data(
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
        print(f"\nPreprocessing complete! Using {len(df)} rows for clustering.")
    else:
        print(f"\nLoading data from {data_path}...")
        df = pd.read_excel(data_path)
        print(f"Loaded {len(df)} rows")
    
    if column_name not in df.columns:
        print(f"[ERROR] Column '{column_name}' not found. Available columns: {df.columns.tolist()}")
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df = df.dropna(subset=[column_name])
    df = df[df[column_name].astype(str).str.strip() != '']
    print(f"Rows with valid text: {len(df)}")
    
    texts = df[column_name].astype(str).tolist()
    
    print(f"\nInitializing BERTopic model (min_topic_size={min_topic_size})...")
    print("This may take a while on first run (downloading model)...")
    print("Note: GPU will be used automatically if available through sentence-transformers")
    topic_model = BERTopic(min_topic_size=min_topic_size, verbose=True)
    
    print("\nFitting BERTopic model...")
    print("This may take several minutes depending on data size...")
    topics, probs = topic_model.fit_transform(texts)
    
    print(f"\nClustering complete!")
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers = list(topics).count(-1)
    print(f"Number of topics found: {n_topics}")
    print(f"Number of outliers: {n_outliers}")
    
    df['topic'] = topics
    df['topic_probability'] = probs
    
    print("\nExtracting topic information...")
    topic_descriptions = {}
    unique_topics = sorted(set(topics))
    
    for topic_id in unique_topics:
        if topic_id == -1:
            topic_descriptions[topic_id] = "Noise/Outliers"
            continue
        
        topic_words = topic_model.get_topic(topic_id)
        if topic_words:
            top_keywords = [word for word, _ in topic_words[:10]]
            keywords_str = ', '.join(top_keywords[:5])
            
            representative_docs = topic_model.get_representative_docs(topic_id)
            if representative_docs:
                rep_text = representative_docs[0]
                if len(rep_text) > 200:
                    rep_text = rep_text[:200] + "..."
            else:
                rep_text = "No representative document available"
            
            topic_description = f"Topic: {keywords_str}\nRepresentative text: {rep_text}"
            topic_descriptions[topic_id] = topic_description
    
    df['topic_description'] = df['topic'].map(topic_descriptions)
    
    print("\nTopic Information:")
    print("-" * 60)
    topic_info = topic_model.get_topic_info()
    print(topic_info)
    
    print("\n" + "="*60)
    print("CLUSTERING SUMMARY")
    print("="*60)
    print(f"\nTotal number of topics: {n_topics}")
    print(f"Number of outliers: {n_outliers}")
    print(f"\nTopic Details:")
    print("-" * 60)
    
    for topic_id in sorted([t for t in unique_topics if t != -1]):
        topic_words = topic_model.get_topic(topic_id)
        if topic_words:
            top_keywords = [word for word, _ in topic_words[:10]]
            topic_count = list(topics).count(topic_id)
            
            representative_docs = topic_model.get_representative_docs(topic_id)
            rep_text = representative_docs[0][:200] + "..." if representative_docs and len(representative_docs[0]) > 200 else (representative_docs[0] if representative_docs else "N/A")
            
            print(f"\nTopic {topic_id}:")
            print(f"Documents: {topic_count}")
            print(f"Top keywords: {', '.join(top_keywords[:8])}")
            print(f"Representative text: {rep_text}")
    
    if -1 in unique_topics:
        print(f"\nNoise Topic (-1):")
        print(f"  Documents: {n_outliers}")
        print(f"  These are outliers that don't fit into any topic")
    
    print("\n" + "="*60)
    print("Saving clustered data...")
    df.to_excel(output_path, index=False)
    print(f"Clustered data saved to {output_path}")
    print("="*60)
    
    return df, topic_model


def main(preprocess_input_path='data/Final_table_results.xlsx',
         preprocess_output_path='data/cleaned_data.xlsx',
         preprocess_label_column='Label',
         preprocess_country_column='Country',
         preprocess_target_country='United States',
         preprocess_valid_labels=None,
         preprocess_filtering_mode='strict',
         preprocess_thresholds=None,
         min_topic_size=10,
         output_path='data/bertopic_clustered_data.xlsx',
         column_name='content_text'):
    """
    Main function to run preprocessing and BERTopic clustering.
    
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
    preprocess_filtering_mode : str, default='strict'
        Preprocessing filtering mode: 'strict', 'loose', or 'none'
    preprocess_thresholds : dict, optional
        Custom thresholds for preprocessing
    min_topic_size : int, default=10
        Minimum size of topics for BERTopic
    output_path : str, default='data/bertopic_clustered_data.xlsx'
        Path to save clustered data
    column_name : str, default='content_text'
        Name of the text column
    """
    print("\n" + "="*80)
    print("BERTOPIC CLUSTERING PIPELINE")
    print("="*80)
    
    df_clustered, topic_model = cluster_with_bertopic(
        data_path=preprocess_output_path,
        column_name=column_name,
        output_path=output_path,
        min_topic_size=min_topic_size,
        run_preprocessing=False,
        preprocess_input_path=preprocess_input_path,
        preprocess_output_path=preprocess_output_path,
        preprocess_label_column=preprocess_label_column,
        preprocess_country_column=preprocess_country_column,
        preprocess_target_country=preprocess_target_country,
        preprocess_valid_labels=preprocess_valid_labels,
        preprocess_filtering_mode=preprocess_filtering_mode,
        preprocess_thresholds=preprocess_thresholds
    )
    
    print("\n" + "="*80)
    print("CLUSTERING COMPLETE!")
    print("="*80)
    print(f"\nOutput file: data/bertopic_clustered_data.xlsx")
    print(f"Total topics found: {len(set(df_clustered['topic'])) - (1 if -1 in df_clustered['topic'].values else 0)}")
    print("="*80)


if __name__ == "__main__":
    main()

