import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from cleaning.preprocess_data import preprocess_data
from clustering.bertopic_clustering import cluster_with_bertopic
from clustering.llm_cluster_analysis import analyze_clusters


def run_full_pipeline(
    # Preprocessing parameters
    preprocess_input_path='data/Final_table_results.xlsx',
    preprocess_output_path='data/cleaned_data.xlsx',
    preprocess_label_column='Label',
    preprocess_country_column='Country',
    preprocess_target_country='United States',
    preprocess_valid_labels=None,
    preprocess_filtering_mode='loose',
    preprocess_thresholds=None,
    column_name='content_text',
    
    # BERTopic parameters
    bertopic_min_topic_size=10,
    bertopic_output_path='data/bertopic_clustered_data.xlsx',
    
    # LLM Analysis parameters
    llm_api_key=None,
    llm_model="gpt-4o-mini",
    llm_output_path='data/llm_analyzed_clusters.xlsx'
):
    """
    Run the complete pipeline: preprocessing -> BERTopic clustering -> LLM analysis.
    
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
    column_name : str, default='content_text'
        Name of the text column
    bertopic_min_topic_size : int, default=10
        Minimum size of topics for BERTopic
    bertopic_output_path : str, default='data/bertopic_clustered_data.xlsx'
        Path to save BERTopic clustered data
    llm_api_key : str, optional
        OpenAI API key. If None, uses OPENAI_API_KEY environment variable
    llm_model : str, default="gpt-4o-mini"
        OpenAI model to use for LLM analysis
    llm_output_path : str, default='data/llm_analyzed_clusters.xlsx'
        Path to save LLM analyzed data
        
    Returns:
    --------
    tuple: (pandas.DataFrame, pandas.DataFrame, pandas.DataFrame)
        Cleaned data, clustered data, and analyzed data DataFrames
    """
    print("\n" + "="*80)
    print("FULL PIPELINE: PREPROCESSING -> BERTOPIC -> LLM ANALYSIS")
    print("="*80)
    
    # Step 1: Preprocessing
    print("\n" + "="*80)
    print("[STEP 1/3] PREPROCESSING")
    print("="*80)
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
    print(f"\n✓ Preprocessing complete! Output: {preprocess_output_path}")
    print(f"  Rows: {len(df_cleaned)}")
    
    # Step 2: BERTopic Clustering
    print("\n" + "="*80)
    print("[STEP 2/3] BERTOPIC CLUSTERING")
    print("="*80)
    df_clustered, topic_model = cluster_with_bertopic(
        data_path=preprocess_output_path,
        column_name=column_name,
        output_path=bertopic_output_path,
        min_topic_size=bertopic_min_topic_size,
        run_preprocessing=False,  # Already preprocessed
        preprocess_input_path=preprocess_input_path,
        preprocess_output_path=preprocess_output_path,
        preprocess_label_column=preprocess_label_column,
        preprocess_country_column=preprocess_country_column,
        preprocess_target_country=preprocess_target_country,
        preprocess_valid_labels=preprocess_valid_labels,
        preprocess_filtering_mode=preprocess_filtering_mode,
        preprocess_thresholds=preprocess_thresholds
    )
    print(f"\n✓ BERTopic clustering complete! Output: {bertopic_output_path}")
    n_topics = len(set(df_clustered['topic'])) - (1 if -1 in df_clustered['topic'].values else 0)
    print(f"  Topics found: {n_topics}")
    
    # Step 3: LLM Analysis
    print("\n" + "="*80)
    print("[STEP 3/3] LLM CLUSTER ANALYSIS")
    print("="*80)
    df_analyzed = analyze_clusters(
        data_path=bertopic_output_path,
        column_name=column_name,
        output_path=llm_output_path,
        api_key=llm_api_key,
        model=llm_model
    )
    print(f"\n✓ LLM analysis complete! Output: {llm_output_path}")
    
    # Final Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nOutput Files:")
    print(f"  1. Cleaned data: {preprocess_output_path}")
    print(f"  2. Clustered data: {bertopic_output_path}")
    print(f"  3. LLM analyzed data: {llm_output_path}")
    print(f"\nSummary:")
    print(f"  - Cleaned rows: {len(df_cleaned)}")
    print(f"  - Topics found: {n_topics}")
    print(f"  - Analyzed clusters: {len([t for t in df_analyzed['topic'].unique() if t != -1])}")
    print("="*80)
    
    return df_cleaned, df_clustered, df_analyzed


def main():
    """
    Main function to run the full pipeline with default parameters.
    """
    run_full_pipeline()


if __name__ == "__main__":
    main()

