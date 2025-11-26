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

def log_hapax_removal(row_num, hapax_value, threshold, text=None):
    """Log hapax ratio-based row removal - comment out the print line to disable hapax logs"""
    words_preview = f" '{_get_first_two_words(text)}'" if text else ""
    print(f"[HAPAX] Row {row_num}{words_preview}: Dropped (Hapax Ratio: {hapax_value:.4f} < {threshold})")  # Comment this line to disable hapax logging

def log_repetition_removal(row_num, repetition_ratio, threshold, top_word, text=None):
    """Log repetition ratio-based row removal - comment out the print line to disable repetition logs"""
    words_preview = f" '{_get_first_two_words(text)}'" if text else ""
    print(f"[REPETITION] Row {row_num}{words_preview}: Dropped (Repetition Ratio: {repetition_ratio:.4f} > {threshold}, top word: '{top_word}')")  # Comment this line to disable repetition logging

def log_stopword_removal(row_num, stopword_ratio, threshold, text=None):
    """Log stopword ratio-based row removal - comment out the print line to disable stopword logs"""
    words_preview = f" '{_get_first_two_words(text)}'" if text else ""
    print(f"[STOPWORD] Row {row_num}{words_preview}: Dropped (Stopword Ratio: {stopword_ratio:.4f} > {threshold})")  # Comment this line to disable stopword logging


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
    
    tokens = [word.lower().strip() for word in text.split() if word.strip()]
    
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
    
    tokens = [word.lower().strip() for word in text.split() if word.strip()]
    
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


### Hapax Ratio Check Functions ###
def calculate_hapax_ratio(text):
    """
    Calculate Hapax Ratio for a given text.
    
    Hapax Ratio = number of words that appear exactly once (hapax legomena) / total number of words
    
    Higher hapax ratio indicates more unique/rare words in the text.
    Lower hapax ratio indicates more repetitive/common words.
    
    Parameters:
    -----------
    text : str
        Text string to calculate hapax ratio for
        
    Returns:
    --------
    float
        Hapax ratio value between 0 and 1, or None if text is empty/invalid
    """
    if pd.isna(text) or text is None:
        return None
    
    text = str(text).strip()
    if not text:
        return None
    
    tokens = [word.lower().strip() for word in text.split() if word.strip()]
    
    if len(tokens) == 0:
        return None
    
    word_counts = {}
    for token in tokens:
        word_counts[token] = word_counts.get(token, 0) + 1
    
    hapax_count = sum(1 for count in word_counts.values() if count == 1)
    
    hapax_ratio = hapax_count / len(tokens)
    
    return hapax_ratio


def filter_by_hapax_ratio(df, column_name='content_text', hapax_threshold=0.3, drop_below=True):
    """
    Filter DataFrame rows based on Hapax Ratio threshold.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data
    column_name : str, default='content_text'
        The name of the column containing text content
    hapax_threshold : float, default=0.3
        Hapax ratio threshold value (between 0 and 1)
    drop_below : bool, default=True
        If True, drop rows with hapax ratio below threshold (low uniqueness).
        If False, drop rows with hapax ratio above threshold (high uniqueness).
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with rows filtered based on hapax ratio threshold
    """
    df_filtered = df.copy()
    
    if column_name not in df_filtered.columns:
        log_info(f"[Warning] Column '{column_name}' not found in DataFrame")
        log_info(f"[INFO] Available columns: {df_filtered.columns.tolist()}")
        return df_filtered
    
    log_info(f"\n[INFO] Filtering rows based on Hapax Ratio threshold ({hapax_threshold})...")
    log_info(f"[INFO] Initial row count: {len(df_filtered)}")
    
    hapax_values = df_filtered[column_name].apply(calculate_hapax_ratio)
    
    if drop_below:
        dropped_rows = df_filtered[hapax_values < hapax_threshold].index
        for row_idx in dropped_rows:
            hapax_val = hapax_values.loc[row_idx]
            if pd.notna(hapax_val):
                text_content = df_filtered.loc[row_idx, column_name]
                log_hapax_removal(row_idx, hapax_val, hapax_threshold, text_content)
        mask = (hapax_values >= hapax_threshold) | (hapax_values.isna())
        dropped_count = (hapax_values < hapax_threshold).sum()
        log_info(f"[INFO] Dropping {dropped_count} rows with Hapax Ratio < {hapax_threshold}")
    else:
        dropped_rows = df_filtered[hapax_values > hapax_threshold].index
        for row_idx in dropped_rows:
            hapax_val = hapax_values.loc[row_idx]
            if pd.notna(hapax_val):
                text_content = df_filtered.loc[row_idx, column_name]
                log_hapax_removal(row_idx, hapax_val, hapax_threshold, text_content)
        mask = (hapax_values <= hapax_threshold) | (hapax_values.isna())
        dropped_count = (hapax_values > hapax_threshold).sum()
        log_info(f"[INFO] Dropping {dropped_count} rows with Hapax Ratio > {hapax_threshold}")
    
    df_filtered = df_filtered[mask].copy()
    
    log_info(f"[INFO] Remaining rows after Hapax Ratio filtering: {len(df_filtered)}")
    
    return df_filtered


### Repetition Ratio Check Functions ###
def calculate_repetition_ratio(text):
    """
    Calculate Repetition Ratio for a given text.
    
    Repetition Ratio = count of most frequent word / total number of words
    
    Higher repetition ratio indicates that one word dominates the text (low quality/repetitive).
    Lower repetition ratio indicates more diverse word distribution.
    
    Parameters:
    -----------
    text : str
        Text string to calculate repetition ratio for
        
    Returns:
    --------
    tuple
        (repetition_ratio, top_word) where:
        - repetition_ratio: float between 0 and 1, or None if text is empty/invalid
        - top_word: str, the most frequent word, or None if text is empty/invalid
    """
    if pd.isna(text) or text is None:
        return None, None
    
    text = str(text).strip()
    if not text:
        return None, None
    
    tokens = [word.lower().strip() for word in text.split() if word.strip()]
    
    if len(tokens) == 0:
        return None, None
    
    word_counts = {}
    for token in tokens:
        word_counts[token] = word_counts.get(token, 0) + 1
    
    if len(word_counts) == 0:
        return None, None
    
    top_word = max(word_counts, key=word_counts.get)
    top_word_count = word_counts[top_word]
    
    repetition_ratio = top_word_count / len(tokens)
    
    return repetition_ratio, top_word


def filter_by_repetition_ratio(df, column_name='content_text', repetition_threshold=0.3, drop_above=True):
    """
    Filter DataFrame rows based on Repetition Ratio threshold.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data
    column_name : str, default='content_text'
        The name of the column containing text content
    repetition_threshold : float, default=0.3
        Repetition ratio threshold value (between 0 and 1)
        If drop_above=True, rows with ratio > threshold are dropped
    drop_above : bool, default=True
        If True, drop rows with repetition ratio above threshold (high repetition).
        If False, drop rows with repetition ratio below threshold (low repetition).
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with rows filtered based on repetition ratio threshold
    """
    df_filtered = df.copy()
    
    if column_name not in df_filtered.columns:
        log_info(f"[Warning] Column '{column_name}' not found in DataFrame")
        log_info(f"[INFO] Available columns: {df_filtered.columns.tolist()}")
        return df_filtered
    
    log_info(f"\n[INFO] Filtering rows based on Repetition Ratio threshold ({repetition_threshold})...")
    log_info(f"[INFO] Initial row count: {len(df_filtered)}")
    
    repetition_data = df_filtered[column_name].apply(calculate_repetition_ratio)
    repetition_ratios = repetition_data.apply(lambda x: x[0] if x[0] is not None else None)
    top_words = repetition_data.apply(lambda x: x[1] if x[1] is not None else None)
    
    if drop_above:
        dropped_rows = df_filtered[repetition_ratios > repetition_threshold].index
        for row_idx in dropped_rows:
            rep_ratio = repetition_ratios.loc[row_idx]
            top_word = top_words.loc[row_idx]
            if pd.notna(rep_ratio):
                text_content = df_filtered.loc[row_idx, column_name]
                log_repetition_removal(row_idx, rep_ratio, repetition_threshold, top_word, text_content)
        mask = (repetition_ratios <= repetition_threshold) | (repetition_ratios.isna())
        dropped_count = (repetition_ratios > repetition_threshold).sum()
        log_info(f"[INFO] Dropping {dropped_count} rows with Repetition Ratio > {repetition_threshold}")
    else:
        dropped_rows = df_filtered[repetition_ratios < repetition_threshold].index
        for row_idx in dropped_rows:
            rep_ratio = repetition_ratios.loc[row_idx]
            top_word = top_words.loc[row_idx]
            if pd.notna(rep_ratio):
                text_content = df_filtered.loc[row_idx, column_name]
                log_repetition_removal(row_idx, rep_ratio, repetition_threshold, top_word, text_content)
        mask = (repetition_ratios >= repetition_threshold) | (repetition_ratios.isna())
        dropped_count = (repetition_ratios < repetition_threshold).sum()
        log_info(f"[INFO] Dropping {dropped_count} rows with Repetition Ratio < {repetition_threshold}")
    
    df_filtered = df_filtered[mask].copy()
    
    log_info(f"[INFO] Remaining rows after Repetition Ratio filtering: {len(df_filtered)}")
    
    return df_filtered


### Stopword Filter Functions ###
_STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
    'have', 'had', 'what', 'said', 'each', 'which', 'their', 'time',
    'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
    'her', 'would', 'make', 'like', 'into', 'him', 'has', 'two',
    'more', 'very', 'after', 'words', 'long', 'than', 'first', 'been',
    'call', 'who', 'oil', 'its', 'now', 'find', 'down', 'day', 'did',
    'get', 'come', 'made', 'may', 'part', 'over', 'new', 'sound',
    'take', 'only', 'little', 'work', 'know', 'place', 'year', 'live',
    'me', 'back', 'give', 'most', 'very', 'after', 'thing', 'our',
    'just', 'name', 'good', 'sentence', 'man', 'think', 'say', 'great',
    'where', 'help', 'through', 'much', 'before', 'line', 'right', 'too',
    'mean', 'old', 'any', 'same', 'tell', 'boy', 'follow', 'came',
    'want', 'show', 'also', 'around', 'form', 'three', 'small', 'set',
    'put', 'end', 'does', 'another', 'well', 'large', 'must', 'big',
    'even', 'such', 'because', 'turn', 'here', 'why', 'ask', 'went',
    'men', 'read', 'need', 'land', 'different', 'home', 'us', 'move',
    'try', 'kind', 'hand', 'picture', 'again', 'change', 'off', 'play',
    'spell', 'air', 'away', 'animal', 'house', 'point', 'page', 'letter',
    'mother', 'answer', 'found', 'study', 'still', 'learn', 'should',
    'america', 'world', 'high', 'every', 'near', 'add', 'food', 'between',
    'own', 'below', 'country', 'plant', 'last', 'school', 'father', 'keep',
    'tree', 'never', 'start', 'city', 'earth', 'eye', 'light', 'thought',
    'head', 'under', 'story', 'saw', 'left', 'don\'t', 'few', 'while',
    'along', 'might', 'close', 'something', 'seem', 'next', 'hard', 'open',
    'example', 'begin', 'life', 'always', 'those', 'both', 'paper', 'together',
    'got', 'group', 'often', 'run', 'important', 'until', 'children', 'side',
    'feet', 'car', 'mile', 'night', 'walk', 'white', 'sea', 'began', 'grow',
    'took', 'river', 'four', 'carry', 'state', 'once', 'book', 'hear',
    'stop', 'without', 'second', 'later', 'miss', 'idea', 'enough', 'eat',
    'face', 'watch', 'far', 'indian', 'really', 'almost', 'let', 'above',
    'girl', 'sometimes', 'mountain', 'cut', 'young', 'talk', 'soon', 'list',
    'song', 'leave', 'family', 'it\'s'
}

def calculate_stopword_ratio(text, stopwords=None):
    """
    Calculate Stopword Ratio for a given text.
    
    Stopword Ratio = number of stopwords / total number of words
    
    Higher stopword ratio indicates text with many common/function words (may be low quality).
    Lower stopword ratio indicates more substantive content words.
    
    Parameters:
    -----------
    text : str
        Text string to calculate stopword ratio for
    stopwords : set, optional
        Set of stopwords to use. If None, uses default English stopwords.
        
    Returns:
    --------
    float
        Stopword ratio value between 0 and 1, or None if text is empty/invalid
    """
    if pd.isna(text) or text is None:
        return None
    
    text = str(text).strip()
    if not text:
        return None
    
    tokens = [word.lower().strip() for word in text.split() if word.strip()]
    
    if len(tokens) == 0:
        return None
    
    if stopwords is None:
        stopwords = _STOPWORDS
    
    stopword_count = sum(1 for token in tokens if token in stopwords)
    
    stopword_ratio = stopword_count / len(tokens)
    
    return stopword_ratio


def filter_by_stopword_ratio(df, column_name='content_text', stopword_threshold=0.5, 
                              stopwords=None, drop_above=True):
    """
    Filter DataFrame rows based on Stopword Ratio threshold.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data
    column_name : str, default='content_text'
        The name of the column containing text content
    stopword_threshold : float, default=0.5
        Stopword ratio threshold value (between 0 and 1)
        If drop_above=True, rows with ratio > threshold are dropped
    stopwords : set, optional
        Set of stopwords to use. If None, uses default English stopwords.
    drop_above : bool, default=True
        If True, drop rows with stopword ratio above threshold (high stopword content).
        If False, drop rows with stopword ratio below threshold (low stopword content).
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with rows filtered based on stopword ratio threshold
    """
    df_filtered = df.copy()
    
    if column_name not in df_filtered.columns:
        log_info(f"[Warning] Column '{column_name}' not found in DataFrame")
        log_info(f"[INFO] Available columns: {df_filtered.columns.tolist()}")
        return df_filtered
    
    log_info(f"\n[INFO] Filtering rows based on Stopword Ratio threshold ({stopword_threshold})...")
    log_info(f"[INFO] Initial row count: {len(df_filtered)}")
    
    stopword_values = df_filtered[column_name].apply(
        lambda x: calculate_stopword_ratio(x, stopwords=stopwords)
    )
    
    if drop_above:
        dropped_rows = df_filtered[stopword_values > stopword_threshold].index
        for row_idx in dropped_rows:
            stopword_val = stopword_values.loc[row_idx]
            if pd.notna(stopword_val):
                text_content = df_filtered.loc[row_idx, column_name]
                log_stopword_removal(row_idx, stopword_val, stopword_threshold, text_content)
        mask = (stopword_values <= stopword_threshold) | (stopword_values.isna())
        dropped_count = (stopword_values > stopword_threshold).sum()
        log_info(f"[INFO] Dropping {dropped_count} rows with Stopword Ratio > {stopword_threshold}")
    else:
        dropped_rows = df_filtered[stopword_values < stopword_threshold].index
        for row_idx in dropped_rows:
            stopword_val = stopword_values.loc[row_idx]
            if pd.notna(stopword_val):
                text_content = df_filtered.loc[row_idx, column_name]
                log_stopword_removal(row_idx, stopword_val, stopword_threshold, text_content)
        mask = (stopword_values >= stopword_threshold) | (stopword_values.isna())
        dropped_count = (stopword_values < stopword_threshold).sum()
        log_info(f"[INFO] Dropping {dropped_count} rows with Stopword Ratio < {stopword_threshold}")
    
    df_filtered = df_filtered[mask].copy()
    
    log_info(f"[INFO] Remaining rows after Stopword Ratio filtering: {len(df_filtered)}")
    
    return df_filtered

