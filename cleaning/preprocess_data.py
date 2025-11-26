import pandas as pd
import os
import re
from .remove_boilerplates import remove_html_tags, remove_boilerplate_patterns
from .lexical_quality_check import filter_by_ttr, filter_by_entropy, filter_by_mtld, filter_by_hapax_ratio, filter_by_repetition_ratio, filter_by_stopword_ratio


# Threshold configurations
LOOSE_THRESHOLDS = {
    "ttr": 0.10,        
    "entropy": 4.0,          
    "mtld": 15.0,           
    "hapax_ratio": 0.08,     
    "repetition_ratio": 0.30,
    "stopword_ratio": 0.50,  
}

STRICT_THRESHOLDS = {
    "ttr": 0.15,              
    "entropy": 4.1,          
    "mtld": 23.0,       
    "hapax_ratio": 0.13,   
    "repetition_ratio": 0.12, 
    "stopword_ratio": 0.35,  
}

def filter_by_label_and_country(df, label_column='Label', country_column='Country',
                                target_country='United States', valid_labels=None):
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
    target_country : str, default='United States'
        The country to filter for (case-insensitive)
    valid_labels : set or list, optional
        Set or list of valid labels to filter for. If None, defaults to 
        {'research materials', 'technical documentation'}
        
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
    
    if valid_labels is None:
        valid_labels = {'research materials', 'technical documentation'}
    else:
        # Convert to set and normalize to lowercase
        valid_labels = {str(label).lower().strip() for label in valid_labels}
    
    print(f"\n[Filter 1/2] Filtering by Country ({target_country} only)...")
    before_country = len(df_filtered)
    target_country_lower = str(target_country).lower().strip()
    df_filtered = df_filtered[df_filtered[country_col].astype(str).str.lower().str.strip() == target_country_lower]
    after_country = len(df_filtered)
    print(f"  Rows before country filter: {before_country}")
    print(f"  Rows after country filter: {after_country}")
    print(f"  Removed: {before_country - after_country} rows")
    
    valid_labels_str = ', '.join(sorted(valid_labels))
    print(f"\n[Filter 2/2] Filtering by Label ({valid_labels_str} only)...")
    before_label = len(df_filtered)
    
    def is_valid_label(label_value):
        """
        Check if label contains only labels from the valid_labels set.
        Returns True if the label contains one or more valid labels and no other labels.
        """
        if pd.isna(label_value):
            return False
        
        label_str = str(label_value).lower().strip()
        
        labels = re.split(r'[,;|\n\r]+', label_str)
        labels = [l.strip() for l in labels if l.strip()]
        
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


def preprocess_data(input_path='data/Final_table_results.xlsx', 
                   output_path='data/cleaned_data.xlsx',
                   label_column='Label', 
                   country_column='Country',
                   target_country='United States',
                   valid_labels=None,
                   filtering_mode='loose',
                   thresholds=None,
                   column_name='content_text',
                   save_output=True):
    """
    Preprocess data by filtering and cleaning.
    
    Parameters:
    -----------
    input_path : str, default='data/Final_table_results.xlsx'
        Path to the input Excel file
    output_path : str, default='data/cleaned_data.xlsx'
        Path to save the cleaned data Excel file
    label_column : str, default='Label'
        Name of the column containing labels
    country_column : str, default='Country'
        Name of the column containing country information
    target_country : str, default='United States'
        The country to filter for (case-insensitive)
    valid_labels : set or list, optional
        Set or list of valid labels to filter for. If None, defaults to 
        {'research materials', 'technical documentation'}
    filtering_mode : str, default='loose'
        Filtering mode: 'strict', 'loose', or 'none'
        - 'strict': Uses STRICT_THRESHOLDS for high-quality data
        - 'loose': Uses LOOSE_THRESHOLDS for broader data retention
        - 'none': Skips quality filtering (only label/country filtering)
    thresholds : dict, optional
        Custom threshold dictionary. If provided, overrides the default thresholds.
        Expected keys: 'ttr', 'entropy', 'mtld', 'hapax_ratio', 'repetition_ratio', 'stopword_ratio'
    column_name : str, default='content_text'
        Name of the column containing text content
    save_output : bool, default=True
        Whether to save the cleaned data to output_path
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned and filtered DataFrame
    """
    print("\n" + "="*60)
    print("DATA PREPROCESSING")
    print("="*60)
    
    print(f"\nLoading Excel file from {input_path}...")
    df_full = pd.read_excel(input_path)
    print(f"Total rows in dataset: {len(df_full)}")
    
    df_filtered = filter_by_label_and_country(df_full, 
                                             label_column=label_column, 
                                             country_column=country_column,
                                             target_country=target_country,
                                             valid_labels=valid_labels)
    
    print(f"\nFiltered dataset: {len(df_filtered)} rows")
    
    if filtering_mode == 'strict':
        df_cleaned = apply_strict_filtering(df_filtered, 
                                           column_name=column_name, 
                                           thresholds=thresholds)
    elif filtering_mode == 'loose':
        df_cleaned = apply_loose_filtering(df_filtered, 
                                          column_name=column_name, 
                                          thresholds=thresholds)
    elif filtering_mode == 'none':
        print("\nSkipping quality filtering (filtering_mode='none')...")
        df_cleaned = df_filtered.copy()
    else:
        raise ValueError(f"Invalid filtering_mode: {filtering_mode}. Must be 'strict', 'loose', or 'none'")
    
    print(f"\nData preprocessing complete!")
    print(f"Final DataFrame shape: {df_cleaned.shape}")
    
    if save_output:
        print(f"\nSaving cleaned data to {output_path}...")
        df_cleaned.to_excel(output_path, index=False)
        print(f"Cleaned data saved successfully to {output_path}")
    
    return df_cleaned