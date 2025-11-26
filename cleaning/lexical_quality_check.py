import pandas as pd
import re


### Logging Functions ###
def log_info(message, row_num=None):
    """Log info messages - comment out the print line to disable info logs"""
    row_info = f"Row {row_num}: " if row_num is not None else ""
    print(f"{row_info}{message}")  # Comment this line to disable info logging

def log_ttr_removal(row_num, ttr_value, threshold):
    """Log TTR-based row removal - comment out the print line to disable TTR logs"""
    print(f"  [TTR] Row {row_num}: Dropped (TTR: {ttr_value:.4f} < {threshold})")  # Comment this line to disable TTR logging


### Lexical Quality Check Functions ###
def calculate_ttr(text):
    """
    Calculate Type-Token Ratio (TTR) for a given text.
    
    TTR = number of unique words (types) / total number of words (tokens)
    
    Parameters:
    -----------
    text : str
        Text string to calculate TTR for
        
    Returns:
    --------
    float
        TTR value between 0 and 1, or None if text is empty/invalid
    """
    if pd.isna(text) or text is None:
        return None
    
    text = str(text).strip()
    if not text:
        return None
    
    tokens = [word.lower() for word in text.split() if word.strip()]
    
    if len(tokens) == 0:
        return None
    
    types = len(set(tokens))
    
    ttr = types / len(tokens)
    
    return ttr


def filter_by_ttr(df, column_name='content_text', ttr_threshold=0.12, drop_below=True):
    """
    Filter DataFrame rows based on Type-Token Ratio (TTR) threshold.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data
    column_name : str, default='content_text'
        The name of the column containing text content
    ttr_threshold : float, default=0.3
        TTR threshold value (between 0 and 1)
    drop_below : bool, default=True
        If True, drop rows with TTR below threshold (low diversity).
        If False, drop rows with TTR above threshold (high diversity).
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with rows filtered based on TTR threshold
    """
    df_filtered = df.copy()
    
    if column_name not in df_filtered.columns:
        log_info(f"[Warning] Column '{column_name}' not found in DataFrame")
        log_info(f"[INFO] Available columns: {df_filtered.columns.tolist()}")
        return df_filtered
    
    log_info(f"\n[INFO] Filtering rows based on TTR threshold ({ttr_threshold})...")
    log_info(f"[INFO] Initial row count: {len(df_filtered)}")
    
    ttr_values = df_filtered[column_name].apply(calculate_ttr)
    
    if drop_below:
        dropped_rows = df_filtered[ttr_values < ttr_threshold].index
        for row_idx in dropped_rows:
            ttr_val = ttr_values.loc[row_idx]
            if pd.notna(ttr_val):
                log_ttr_removal(row_idx, ttr_val, ttr_threshold)
        mask = (ttr_values >= ttr_threshold) | (ttr_values.isna())
        dropped_count = (ttr_values < ttr_threshold).sum()
        log_info(f"[INFO] Dropping {dropped_count} rows with TTR < {ttr_threshold}")
    else:
        dropped_rows = df_filtered[ttr_values > ttr_threshold].index
        for row_idx in dropped_rows:
            ttr_val = ttr_values.loc[row_idx]
            if pd.notna(ttr_val):
                log_ttr_removal(row_idx, ttr_val, ttr_threshold)
        mask = (ttr_values <= ttr_threshold) | (ttr_values.isna())
        dropped_count = (ttr_values > ttr_threshold).sum()
        log_info(f"[INFO] Dropping {dropped_count} rows with TTR > {ttr_threshold}")
    
    df_filtered = df_filtered[mask].copy()
    
    log_info(f"[INFO] Remaining rows after TTR filtering: {len(df_filtered)}")
    
    return df_filtered

