import pandas as pd
import os
import re
from remove_boilerplates import remove_html_tags, remove_boilerplate_patterns
from lexical_quality_check import filter_by_ttr, filter_by_entropy, filter_by_mtld, filter_by_hapax_ratio, filter_by_repetition_ratio, filter_by_stopword_ratio


# Threshold configurations
LOOSE_THRESHOLDS = {
    "ttr": 0.12,
    "entropy": 4.5,
    "mtld": 20.0,
    "hapax_ratio": 0.10,
    "repetition_ratio": 0.25,
    "stopword_ratio": 0.70,
}

STRICT_THRESHOLDS = {
    "ttr": 0.15,
    "entropy": 5.0,
    "mtld": 30.0,
    "hapax_ratio": 0.15,
    "repetition_ratio": 0.15,
    "stopword_ratio": 0.65,
}


def filter_by_label_and_country(df, label_column='Label', country_column='Country'):
    """
    Filter DataFrame by Label and Country criteria.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to filter
    label_column : str, default='Label'
        The name of the column containing labels
    country_column : str, default='Country'
        The name of the column containing country information
        
    Returns:
    --------
    pandas.DataFrame
        Filtered DataFrame with only rows matching the criteria
    """
    df_filtered = df.copy()
    initial_count = len(df_filtered)
    
    print("\n" + "="*60)
    print("Filtering by Label and Country...")
    print("="*60)
    
    label_col = None
    country_col = None
    
    for col in df_filtered.columns:
        if col.lower() == label_column.lower():
            label_col = col
        if col.lower() == country_column.lower():
            country_col = col
    
    if label_col is None:
        print(f"[WARNING] Label column '{label_column}' not found. Available columns: {df_filtered.columns.tolist()}")
        print("[INFO] Attempting to find label column...")

        for col in df_filtered.columns:
            if 'label' in col.lower():
                label_col = col
                print(f"[INFO] Using column '{col}' as label column")
                break
    
    if country_col is None:
        print(f"[WARNING] Country column '{country_column}' not found. Available columns: {df_filtered.columns.tolist()}")
        print("[INFO] Attempting to find country column...")

        for col in df_filtered.columns:
            if 'country' in col.lower():
                country_col = col
                print(f"[INFO] Using column '{col}' as country column")
                break
    
    if label_col is None or country_col is None:
        raise ValueError(f"Required columns not found. Label: {label_col}, Country: {country_col}")
    
    print(f"\nUsing columns: Label='{label_col}', Country='{country_col}'")
    
    print(f"\n[Filter 1/2] Filtering by Country (United States only)...")
    before_country = len(df_filtered)
    df_filtered = df_filtered[df_filtered[country_col].astype(str).str.lower().str.strip() == 'united states']
    after_country = len(df_filtered)
    print(f"  Rows before country filter: {before_country}")
    print(f"  Rows after country filter: {after_country}")
    print(f"  Removed: {before_country - after_country} rows")
    
    print(f"\n[Filter 2/2] Filtering by Label (research materials and/or technical documentation only)...")
    before_label = len(df_filtered)
    
    def is_valid_label(label_value):
        """
        Check if label contains only 'research materials' and/or 'technical documentation'.
        Returns True if the label contains one or both of these and no other labels.
        """
        if pd.isna(label_value):
            return False
        
        label_str = str(label_value).lower().strip()
        
        labels = re.split(r'[,;|\n\r]+', label_str)
        labels = [l.strip() for l in labels if l.strip()]
        
        valid_labels = {'research materials', 'technical documentation'}
        
        for label in labels:
            if label not in valid_labels:
                return False
        
        return len(labels) > 0
    
    df_filtered = df_filtered[df_filtered[label_col].apply(is_valid_label)]
    after_label = len(df_filtered)
    
    print(f"  Rows before label filter: {before_label}")
    print(f"  Rows after label filter: {after_label}")
    print(f"  Removed: {before_label - after_label} rows")
    
    final_count = len(df_filtered)
    print("\n" + "="*60)
    print(f"Label and Country filtering complete!")
    print(f"Initial rows: {initial_count}")
    print(f"Final rows: {final_count}")
    print(f"Rows removed: {initial_count - final_count} ({(initial_count - final_count)/initial_count*100:.2f}%)")
    print("="*60)
    
    return df_filtered


def apply_strict_filtering(df, column_name='content_text', thresholds=None):
    """
    Apply strict filtering with stricter thresholds for high-quality data.
    
    This function applies all cleaning and quality checks with more aggressive thresholds
    to ensure only high-quality content remains.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to filter
    column_name : str, default='content_text'
        The name of the column containing text content
    thresholds : dict, optional
        Dictionary of threshold values. If None, uses STRICT_THRESHOLDS.
        Expected keys: 'ttr', 'entropy', 'mtld', 'hapax_ratio', 'repetition_ratio', 'stopword_ratio'
        
    Returns:
    --------
    pandas.DataFrame
        Filtered DataFrame with strict quality standards
    """
    if thresholds is None:
        thresholds = STRICT_THRESHOLDS
    
    print("\n" + "="*60)
    print("Applying STRICT filtering...")
    print("="*60)
    
    df_cleaned = df.copy()
    initial_count = len(df_cleaned)
    
    print("\n[Step 1/8] Removing HTML tags...")
    df_cleaned = remove_html_tags(df_cleaned, column_name=column_name)
    
    print("\n[Step 2/8] Removing boilerplate patterns...")
    df_cleaned = remove_boilerplate_patterns(df_cleaned, column_name=column_name)
    
    print(f"\n[Step 3/8] Filtering by TTR (threshold: {thresholds['ttr']})...")
    df_cleaned = filter_by_ttr(df_cleaned, column_name=column_name, 
                               ttr_threshold=thresholds['ttr'], drop_below=True)
    
    print(f"\n[Step 4/8] Filtering by Entropy (threshold: {thresholds['entropy']})...")
    df_cleaned = filter_by_entropy(df_cleaned, column_name=column_name, 
                                   entropy_threshold=thresholds['entropy'], drop_below=True)
    
    print(f"\n[Step 5/8] Filtering by MTLD (threshold: {thresholds['mtld']})...")
    df_cleaned = filter_by_mtld(df_cleaned, column_name=column_name, 
                                mtld_threshold=thresholds['mtld'], 
                                ttr_threshold=0.72, min_words=10, drop_below=True)
    
    print(f"\n[Step 6/8] Filtering by Hapax Ratio (threshold: {thresholds['hapax_ratio']})...")
    df_cleaned = filter_by_hapax_ratio(df_cleaned, column_name=column_name, 
                                      hapax_threshold=thresholds['hapax_ratio'], drop_below=True)
    
    print(f"\n[Step 7/8] Filtering by Repetition Ratio (threshold: {thresholds['repetition_ratio']})...")
    df_cleaned = filter_by_repetition_ratio(df_cleaned, column_name=column_name, 
                                           repetition_threshold=thresholds['repetition_ratio'], 
                                           drop_above=True)
    
    print(f"\n[Step 8/8] Filtering by Stopword Ratio (threshold: {thresholds['stopword_ratio']})...")
    df_cleaned = filter_by_stopword_ratio(df_cleaned, column_name=column_name, 
                                         stopword_threshold=thresholds['stopword_ratio'], 
                                         drop_above=True)
    
    final_count = len(df_cleaned)
    print("\n" + "="*60)
    print(f"STRICT filtering complete!")
    print(f"Initial rows: {initial_count}")
    print(f"Final rows: {final_count}")
    print(f"Rows removed: {initial_count - final_count} ({(initial_count - final_count)/initial_count*100:.2f}%)")
    print("="*60)
    
    return df_cleaned


def apply_loose_filtering(df, column_name='content_text', thresholds=None):
    """
    Apply loose filtering with more lenient thresholds for broader data retention.
    
    This function applies all cleaning and quality checks with more lenient thresholds
    to retain more data while still removing obvious low-quality content.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to filter
    column_name : str, default='content_text'
        The name of the column containing text content
    thresholds : dict, optional
        Dictionary of threshold values. If None, uses LOOSE_THRESHOLDS.
        Expected keys: 'ttr', 'entropy', 'mtld', 'hapax_ratio', 'repetition_ratio', 'stopword_ratio'
        
    Returns:
    --------
    pandas.DataFrame
        Filtered DataFrame with loose quality standards
    """
    if thresholds is None:
        thresholds = LOOSE_THRESHOLDS
    
    print("\n" + "="*60)
    print("Applying LOOSE filtering...")
    print("="*60)
    
    df_cleaned = df.copy()
    initial_count = len(df_cleaned)
    
    print("\n[Step 1/8] Removing HTML tags...")
    df_cleaned = remove_html_tags(df_cleaned, column_name=column_name)
    
    print("\n[Step 2/8] Removing boilerplate patterns...")
    df_cleaned = remove_boilerplate_patterns(df_cleaned, column_name=column_name)
    
    print(f"\n[Step 3/8] Filtering by TTR (threshold: {thresholds['ttr']})...")
    df_cleaned = filter_by_ttr(df_cleaned, column_name=column_name, 
                               ttr_threshold=thresholds['ttr'], drop_below=True)
    
    print(f"\n[Step 4/8] Filtering by Entropy (threshold: {thresholds['entropy']})...")
    df_cleaned = filter_by_entropy(df_cleaned, column_name=column_name, 
                                   entropy_threshold=thresholds['entropy'], drop_below=True)
    
    print(f"\n[Step 5/8] Filtering by MTLD (threshold: {thresholds['mtld']})...")
    df_cleaned = filter_by_mtld(df_cleaned, column_name=column_name, 
                                mtld_threshold=thresholds['mtld'], 
                                ttr_threshold=0.72, min_words=5, drop_below=True)
    
    print(f"\n[Step 6/8] Filtering by Hapax Ratio (threshold: {thresholds['hapax_ratio']})...")
    df_cleaned = filter_by_hapax_ratio(df_cleaned, column_name=column_name, 
                                      hapax_threshold=thresholds['hapax_ratio'], drop_below=True)
    
    print(f"\n[Step 7/8] Filtering by Repetition Ratio (threshold: {thresholds['repetition_ratio']})...")
    df_cleaned = filter_by_repetition_ratio(df_cleaned, column_name=column_name, 
                                           repetition_threshold=thresholds['repetition_ratio'], 
                                           drop_above=True)
    
    print(f"\n[Step 8/8] Filtering by Stopword Ratio (threshold: {thresholds['stopword_ratio']})...")
    df_cleaned = filter_by_stopword_ratio(df_cleaned, column_name=column_name, 
                                         stopword_threshold=thresholds['stopword_ratio'], 
                                         drop_above=True)
    
    final_count = len(df_cleaned)
    print("\n" + "="*60)
    print(f"LOOSE filtering complete!")
    print(f"Initial rows: {initial_count}")
    print(f"Final rows: {final_count}")
    print(f"Rows removed: {initial_count - final_count} ({(initial_count - final_count)/initial_count*100:.2f}%)")
    print("="*60)
    
    return df_cleaned


if __name__ == '__main__':
    data_path = os.path.join('data', 'Final_table_results.xlsx')
    
    print("Loading Excel file...")
    df_full = pd.read_excel(data_path)
    
    print(f"Total rows in dataset: {len(df_full)}")
    
    # Filter by Label and Country instead of random sampling
    df_filtered = filter_by_label_and_country(df_full, label_column='Label', country_column='Country')
    
    print(f"\nFiltered dataset: {len(df_filtered)} rows")
    
    # Apply strict filtering (uses STRICT_THRESHOLDS by default)
    # df_cleaned = apply_strict_filtering(df_filtered, column_name='content_text')
    
    # Apply loose filtering (uses LOOSE_THRESHOLDS by default)
    df_cleaned = apply_loose_filtering(df_filtered, column_name='content_text')
    
    # Or use custom thresholds
    # custom_thresholds = {
    #     "ttr": 0.13,
    #     "entropy": 4.0,
    #     "mtld": 25.0,
    #     "hapax_ratio": 0.12,
    #     "repetition_ratio": 0.20,
    #     "stopword_ratio": 0.68,
    # }
    # df_cleaned = apply_loose_filtering(df_sample, column_name='content_text', thresholds=custom_thresholds)
    
    print(f"\nData preprocessing complete!")
    print(f"Final DataFrame shape: {df_cleaned.shape}")
    
    output_path = os.path.join('data', 'cleaned_data.xlsx')
    print(f"\nSaving cleaned data to {output_path}...")
    df_cleaned.to_excel(output_path, index=False)
    print(f"Cleaned data saved successfully to {output_path}")