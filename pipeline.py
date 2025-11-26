import os
import sys

sys.path.append(os.path.dirname(__file__))
from cleaning.preprocess_data import preprocess_data
from clustering.bertopic_clustering import cluster_with_bertopic
from analysis.llm_cluster_analysis import analyze_clusters


def run_full_pipeline(
    # Preprocessing parameters
    preprocess_input_path='data/Final_table_results.xlsx',
    preprocess_output_path='data/cleaned_data.xlsx',
    preprocess_label_column='Label',
    preprocess_country_column='Country',
    preprocess_target_country='United States',
    preprocess_valid_labels=None,
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
    Runs both LOOSE and STRICT filtering modes and saves outputs separately.
    
    Parameters:
    -----------
    preprocess_input_path : str, default='data/Final_table_results.xlsx'
        Path to input data for preprocessing
    preprocess_output_path : str, default='data/cleaned_data.xlsx'
        Base path to save cleaned data (will add _loose/_strict suffix)
    preprocess_label_column : str, default='Label'
        Name of the column containing labels
    preprocess_country_column : str, default='Country'
        Name of the column containing country information
    preprocess_target_country : str, default='United States'
        The country to filter for (case-insensitive)
    preprocess_valid_labels : set or list, optional
        Set or list of valid labels to filter for. If None, defaults to 
        {'research materials', 'technical documentation'}
    preprocess_thresholds : dict, optional
        Custom thresholds for preprocessing
    column_name : str, default='content_text'
        Name of the text column
    bertopic_min_topic_size : int, default=10
        Minimum size of topics for BERTopic
    bertopic_output_path : str, default='data/bertopic_clustered_data.xlsx'
        Base path to save BERTopic clustered data (will add _loose/_strict suffix)
    llm_api_key : str, optional
        OpenAI API key. If None, uses OPENAI_API_KEY environment variable
    llm_model : str, default="gpt-4o-mini"
        OpenAI model to use for LLM analysis
    llm_output_path : str, default='data/llm_analyzed_clusters.xlsx'
        Base path to save LLM analyzed data (will add _loose/_strict suffix)
        
    Returns:
    --------
    dict
        Dictionary with 'loose' and 'strict' keys, each containing:
        (cleaned_df, clustered_df, analyzed_df) tuples
    """
    import os
    
    print("\n" + "="*80)
    print("FULL PIPELINE: PREPROCESSING -> BERTOPIC -> LLM ANALYSIS")
    print("Running both LOOSE and STRICT filtering modes...")
    print("="*80)
    
    results = {}
    
    for mode in ['loose', 'strict']:
        print("\n" + "="*80)
        print(f"PROCESSING MODE: {mode.upper()}")
        print("="*80)
        
        # Generate file paths with suffix
        base_name = os.path.splitext(preprocess_output_path)[0]
        ext = os.path.splitext(preprocess_output_path)[1]
        mode_preprocess_output = f"{base_name}_{mode}{ext}"
        
        base_name = os.path.splitext(bertopic_output_path)[0]
        ext = os.path.splitext(bertopic_output_path)[1]
        mode_bertopic_output = f"{base_name}_{mode}{ext}"
        
        base_name = os.path.splitext(llm_output_path)[0]
        ext = os.path.splitext(llm_output_path)[1]
        mode_llm_output = f"{base_name}_{mode}{ext}"
        
        # Step 1: Preprocessing
        print("\n" + "="*80)
        print(f"[STEP 1/3] PREPROCESSING ({mode.upper()})")
        print("="*80)
        df_cleaned = preprocess_data(
            input_path=preprocess_input_path,
            output_path=mode_preprocess_output,
            label_column=preprocess_label_column,
            country_column=preprocess_country_column,
            target_country=preprocess_target_country,
            valid_labels=preprocess_valid_labels,
            filtering_mode=mode,
            thresholds=preprocess_thresholds,
            column_name=column_name,
            save_output=True
        )
        print(f"\n✓ Preprocessing ({mode}) complete! Output: {mode_preprocess_output}")
        print(f"  Rows: {len(df_cleaned)}")
        
        # Step 2: BERTopic Clustering
        print("\n" + "="*80)
        print(f"[STEP 2/3] BERTOPIC CLUSTERING ({mode.upper()})")
        print("="*80)
        df_clustered, topic_model = cluster_with_bertopic(
            data_path=mode_preprocess_output,
            column_name=column_name,
            output_path=mode_bertopic_output,
            min_topic_size=bertopic_min_topic_size,
            run_preprocessing=False,  # Already preprocessed
            preprocess_input_path=preprocess_input_path,
            preprocess_output_path=mode_preprocess_output,
            preprocess_label_column=preprocess_label_column,
            preprocess_country_column=preprocess_country_column,
            preprocess_target_country=preprocess_target_country,
            preprocess_valid_labels=preprocess_valid_labels,
            preprocess_filtering_mode=mode,
            preprocess_thresholds=preprocess_thresholds
        )
        print(f"\n✓ BERTopic clustering ({mode}) complete! Output: {mode_bertopic_output}")
        n_topics = len(set(df_clustered['topic'])) - (1 if -1 in df_clustered['topic'].values else 0)
        print(f"  Topics found: {n_topics}")
        
        # Step 3: LLM Analysis
        print("\n" + "="*80)
        print(f"[STEP 3/3] LLM CLUSTER ANALYSIS ({mode.upper()})")
        print("="*80)
        df_analyzed = analyze_clusters(
            data_path=mode_bertopic_output,
            column_name=column_name,
            output_path=mode_llm_output,
            api_key=llm_api_key,
            model=llm_model
        )
        print(f"\n✓ LLM analysis ({mode}) complete! Output: {mode_llm_output}")
        
        results[mode] = (df_cleaned, df_clustered, df_analyzed)
    
    # Final Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nOutput Files:")
    print(f"\nLOOSE Mode:")
    print(f"  1. Cleaned data: {os.path.splitext(preprocess_output_path)[0]}_loose{os.path.splitext(preprocess_output_path)[1]}")
    print(f"  2. Clustered data: {os.path.splitext(bertopic_output_path)[0]}_loose{os.path.splitext(bertopic_output_path)[1]}")
    print(f"  3. LLM analyzed data: {os.path.splitext(llm_output_path)[0]}_loose{os.path.splitext(llm_output_path)[1]}")
    print(f"\nSTRICT Mode:")
    print(f"  1. Cleaned data: {os.path.splitext(preprocess_output_path)[0]}_strict{os.path.splitext(preprocess_output_path)[1]}")
    print(f"  2. Clustered data: {os.path.splitext(bertopic_output_path)[0]}_strict{os.path.splitext(bertopic_output_path)[1]}")
    print(f"  3. LLM analyzed data: {os.path.splitext(llm_output_path)[0]}_strict{os.path.splitext(llm_output_path)[1]}")
    
    print(f"\nSummary:")
    for mode in ['loose', 'strict']:
        df_cleaned, df_clustered, df_analyzed = results[mode]
        n_topics = len(set(df_clustered['topic'])) - (1 if -1 in df_clustered['topic'].values else 0)
        n_analyzed = len([t for t in df_analyzed['topic'].unique() if t != -1])
        print(f"\n{mode.upper()} Mode:")
        print(f"  - Cleaned rows: {len(df_cleaned)}")
        print(f"  - Topics found: {n_topics}")
        print(f"  - Analyzed clusters: {n_analyzed}")
    print("="*80)
    
    return results


def main():
    """
    Main function to run the full pipeline with default parameters.
    """
    run_full_pipeline()


if __name__ == "__main__":
    main()

