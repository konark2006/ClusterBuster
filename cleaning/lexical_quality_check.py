import pandas as pd
import re
import math


### Logging Functions ###
def log_info(message, row_num=None):
    """Log info messages - comment out the print line to disable info logs"""
    row_info = f"Row {row_num}: " if row_num is not None else ""
    print(f"{row_info}{message}")  # Comment this line to disable info logging

def log_ttr_removal(row_num, ttr_value, threshold):
    """Log TTR-based row removal - comment out the print line to disable TTR logs"""
    print(f"[TTR] Row {row_num}: Dropped (TTR: {ttr_value:.4f} < {threshold})")  # Comment this line to disable TTR logging

def log_entropy_removal(row_num, entropy_value, threshold):
    """Log entropy-based row removal - comment out the print line to disable entropy logs"""
    print(f"[ENTROPY] Row {row_num}: Dropped (Entropy: {entropy_value:.4f} < {threshold})")  # Comment this line to disable entropy logging


### TTR Check Functions ###
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


### Entropy Check Functions ###
def calculate_entropy(text):
    """
    Calculate Shannon entropy for a given text (character-level).
    
    Entropy = -Σ(p(x) * log2(p(x)))
    where p(x) is the probability of character x occurring in the text.
    
    Higher entropy indicates more randomness/diversity in character distribution.
    Lower entropy indicates repetitive or predictable text.
    
    Parameters:
    -----------
    text : str
        Text string to calculate entropy for
        
    Returns:
    --------
    float
        Entropy value (bits per character), or None if text is empty/invalid
    """
    if pd.isna(text) or text is None:
        return None
    
    text = str(text).strip()
    if not text:
        return None
    
    char_counts = {}
    for char in text:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    text_length = len(text)
    entropy = 0.0
    
    for count in char_counts.values():
        if count > 0:
            probability = count / text_length
            entropy -= probability * math.log2(probability)
    
    return entropy


def filter_by_entropy(df, column_name='content_text', entropy_threshold=3.0, drop_below=True):
    """
    Filter DataFrame rows based on entropy threshold.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data
    column_name : str, default='content_text'
        The name of the column containing text content
    entropy_threshold : float, default=3.0
        Entropy threshold value (typically between 0 and ~8 for text)
    drop_below : bool, default=True
        If True, drop rows with entropy below threshold (low randomness/diversity).
        If False, drop rows with entropy above threshold (high randomness/diversity).
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with rows filtered based on entropy threshold
    """
    df_filtered = df.copy()
    
    if column_name not in df_filtered.columns:
        log_info(f"[Warning] Column '{column_name}' not found in DataFrame")
        log_info(f"[INFO] Available columns: {df_filtered.columns.tolist()}")
        return df_filtered
    
    log_info(f"\n[INFO] Filtering rows based on entropy threshold ({entropy_threshold})...")
    log_info(f"[INFO] Initial row count: {len(df_filtered)}")
    
    entropy_values = df_filtered[column_name].apply(calculate_entropy)
    
    if drop_below:
        dropped_rows = df_filtered[entropy_values < entropy_threshold].index
        for row_idx in dropped_rows:
            entropy_val = entropy_values.loc[row_idx]
            if pd.notna(entropy_val):
                log_entropy_removal(row_idx, entropy_val, entropy_threshold)
        mask = (entropy_values >= entropy_threshold) | (entropy_values.isna())
        dropped_count = (entropy_values < entropy_threshold).sum()
        log_info(f"[INFO] Dropping {dropped_count} rows with entropy < {entropy_threshold}")
    else:
        dropped_rows = df_filtered[entropy_values > entropy_threshold].index
        for row_idx in dropped_rows:
            entropy_val = entropy_values.loc[row_idx]
            if pd.notna(entropy_val):
                log_entropy_removal(row_idx, entropy_val, entropy_threshold)
        mask = (entropy_values <= entropy_threshold) | (entropy_values.isna())
        dropped_count = (entropy_values > entropy_threshold).sum()
        log_info(f"[INFO] Dropping {dropped_count} rows with entropy > {entropy_threshold}")
    
    df_filtered = df_filtered[mask].copy()
    
    log_info(f"[INFO] Remaining rows after entropy filtering: {len(df_filtered)}")
    
    return df_filtered

