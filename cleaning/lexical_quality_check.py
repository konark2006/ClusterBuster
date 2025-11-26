import pandas as pd
import re
import math


### Logging Functions ###
def log_info(message, row_num=None):
    """Log info messages - comment out the print line to disable info logs"""
    row_info = f"Row {row_num}: " if row_num is not None else ""
    print(f"{row_info}{message}")  # Comment this line to disable info logging

def _get_first_two_words(text):
    """Extract first 2 words from text for logging"""
    if pd.isna(text) or text is None:
        return ""
    text_str = str(text).strip()
    words = text_str.split()[:2]
    return " ".join(words) if words else ""

def log_ttr_removal(row_num, ttr_value, threshold, text=None):
    """Log TTR-based row removal - comment out the print line to disable TTR logs"""
    words_preview = f" '{_get_first_two_words(text)}'" if text else ""
    print(f"[TTR] Row {row_num}{words_preview}: Dropped (TTR: {ttr_value:.4f} < {threshold})")  # Comment this line to disable TTR logging

def log_entropy_removal(row_num, entropy_value, threshold, text=None):
    """Log entropy-based row removal - comment out the print line to disable entropy logs"""
    words_preview = f" '{_get_first_two_words(text)}'" if text else ""
    print(f"[ENTROPY] Row {row_num}{words_preview}: Dropped (Entropy: {entropy_value:.4f} < {threshold})")  # Comment this line to disable entropy logging

def log_mtld_removal(row_num, mtld_value, threshold, text=None):
    """Log MTLD-based row removal - comment out the print line to disable MTLD logs"""
    words_preview = f" '{_get_first_two_words(text)}'" if text else ""
    print(f"[MTLD] Row {row_num}{words_preview}: Dropped (MTLD: {mtld_value:.4f} < {threshold})")  # Comment this line to disable MTLD logging


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
                text_content = df_filtered.loc[row_idx, column_name]
                log_ttr_removal(row_idx, ttr_val, ttr_threshold, text_content)
        mask = (ttr_values >= ttr_threshold) | (ttr_values.isna())
        dropped_count = (ttr_values < ttr_threshold).sum()
        log_info(f"[INFO] Dropping {dropped_count} rows with TTR < {ttr_threshold}")
    else:
        dropped_rows = df_filtered[ttr_values > ttr_threshold].index
        for row_idx in dropped_rows:
            ttr_val = ttr_values.loc[row_idx]
            if pd.notna(ttr_val):
                text_content = df_filtered.loc[row_idx, column_name]
                log_ttr_removal(row_idx, ttr_val, ttr_threshold, text_content)
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
                text_content = df_filtered.loc[row_idx, column_name]
                log_entropy_removal(row_idx, entropy_val, entropy_threshold, text_content)
        mask = (entropy_values >= entropy_threshold) | (entropy_values.isna())
        dropped_count = (entropy_values < entropy_threshold).sum()
        log_info(f"[INFO] Dropping {dropped_count} rows with entropy < {entropy_threshold}")
    else:
        dropped_rows = df_filtered[entropy_values > entropy_threshold].index
        for row_idx in dropped_rows:
            entropy_val = entropy_values.loc[row_idx]
            if pd.notna(entropy_val):
                text_content = df_filtered.loc[row_idx, column_name]
                log_entropy_removal(row_idx, entropy_val, entropy_threshold, text_content)
        mask = (entropy_values <= entropy_threshold) | (entropy_values.isna())
        dropped_count = (entropy_values > entropy_threshold).sum()
        log_info(f"[INFO] Dropping {dropped_count} rows with entropy > {entropy_threshold}")
    
    df_filtered = df_filtered[mask].copy()
    
    log_info(f"[INFO] Remaining rows after entropy filtering: {len(df_filtered)}")
    
    return df_filtered


### MTLD Check Functions ###
def calculate_mtld(text, ttr_threshold=0.72, min_words=10):
    """
    Calculate Measure of Textual Lexical Diversity (MTLD) for a given text.
    
    MTLD measures the average length of sequential word strings that maintain
    a TTR above a threshold. It's more stable than TTR for longer texts.
    
    Algorithm:
    1. Move forward through text, calculating TTR for each segment
    2. When TTR drops below threshold, that's a "factor"
    3. Continue until end of text
    4. MTLD_forward = total words / number of factors
    5. Repeat backwards
    6. MTLD = average of forward and backward
    
    Parameters:
    -----------
    text : str
        Text string to calculate MTLD for
    ttr_threshold : float, default=0.72
        TTR threshold for factor calculation (typically 0.72)
    min_words : int, default=10
        Minimum number of words required to calculate MTLD
        
    Returns:
    --------
    float
        MTLD value, or None if text is empty/invalid or has fewer than min_words
    """
    if pd.isna(text) or text is None:
        return None
    
    text = str(text).strip()
    if not text:
        return None
    
    tokens = [word.lower() for word in text.split() if word.strip()]
    
    if len(tokens) < min_words:
        return None
    
    def calculate_mtld_direction(word_list, threshold):
        """Calculate MTLD for one direction (forward or backward)"""
        if len(word_list) == 0:
            return None
        
        factors = 0
        current_pos = 0
        
        while current_pos < len(word_list):
            seen_types = set()
            segment_length = 0
            factor_found = False
            
            for i in range(current_pos, len(word_list)):
                seen_types.add(word_list[i])
                segment_length += 1
                
                if segment_length > 0:
                    current_ttr = len(seen_types) / segment_length
                    
                    if current_ttr < threshold:
                        factors += 1
                        current_pos = i + 1  # Next segment starts after position i
                        factor_found = True
                        break
            
            if not factor_found:
                if segment_length > 0:
                    factors += 1
                break
        
        if factors == 0:
            return None
        
        return len(word_list) / factors
    
    mtld_forward = calculate_mtld_direction(tokens, ttr_threshold)    
    mtld_backward = calculate_mtld_direction(tokens[::-1], ttr_threshold)
    
    if mtld_forward is not None and mtld_backward is not None:
        return (mtld_forward + mtld_backward) / 2.0
    elif mtld_forward is not None:
        return mtld_forward
    elif mtld_backward is not None:
        return mtld_backward
    else:
        return None


def filter_by_mtld(df, column_name='content_text', mtld_threshold=50.0, 
                   ttr_threshold=0.72, min_words=10, drop_below=True):
    """
    Filter DataFrame rows based on MTLD (Measure of Textual Lexical Diversity) threshold.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data
    column_name : str, default='content_text'
        The name of the column containing text content
    mtld_threshold : float, default=50.0
        MTLD threshold value (higher values indicate more diversity)
    ttr_threshold : float, default=0.72
        TTR threshold used in MTLD calculation (typically 0.72)
    min_words : int, default=10
        Minimum number of words required to calculate MTLD
    drop_below : bool, default=True
        If True, drop rows with MTLD below threshold (low diversity).
        If False, drop rows with MTLD above threshold (high diversity).
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with rows filtered based on MTLD threshold
    """
    df_filtered = df.copy()
    
    if column_name not in df_filtered.columns:
        log_info(f"[Warning] Column '{column_name}' not found in DataFrame")
        log_info(f"[INFO] Available columns: {df_filtered.columns.tolist()}")
        return df_filtered
    
    log_info(f"\n[INFO] Filtering rows based on MTLD threshold ({mtld_threshold})...")
    log_info(f"[INFO] Initial row count: {len(df_filtered)}")
    log_info(f"[INFO] Using TTR threshold: {ttr_threshold}, Min words: {min_words}")
    
    mtld_values = df_filtered[column_name].apply(
        lambda x: calculate_mtld(x, ttr_threshold=ttr_threshold, min_words=min_words)
    )
    
    if drop_below:
        dropped_rows = df_filtered[mtld_values < mtld_threshold].index
        for row_idx in dropped_rows:
            mtld_val = mtld_values.loc[row_idx]
            if pd.notna(mtld_val):
                text_content = df_filtered.loc[row_idx, column_name]
                log_mtld_removal(row_idx, mtld_val, mtld_threshold, text_content)
        mask = (mtld_values >= mtld_threshold) | (mtld_values.isna())
        dropped_count = (mtld_values < mtld_threshold).sum()
        log_info(f"[INFO] Dropping {dropped_count} rows with MTLD < {mtld_threshold}")
    else:
        dropped_rows = df_filtered[mtld_values > mtld_threshold].index
        for row_idx in dropped_rows:
            mtld_val = mtld_values.loc[row_idx]
            if pd.notna(mtld_val):
                text_content = df_filtered.loc[row_idx, column_name]
                log_mtld_removal(row_idx, mtld_val, mtld_threshold, text_content)
        mask = (mtld_values <= mtld_threshold) | (mtld_values.isna())
        dropped_count = (mtld_values > mtld_threshold).sum()
        log_info(f"[INFO] Dropping {dropped_count} rows with MTLD > {mtld_threshold}")
    
    df_filtered = df_filtered[mask].copy()
    
    log_info(f"[INFO] Remaining rows after MTLD filtering: {len(df_filtered)}")
    
    return df_filtered

