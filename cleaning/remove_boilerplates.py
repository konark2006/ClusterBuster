import pandas as pd
from bs4 import BeautifulSoup
import re


### Logging Functions ###
def log_html_removal(original, cleaned, row_num=None):
    """Log HTML removal - comment out the print line to disable HTML logs"""
    if original != cleaned and '<' in original and '>' in original:
        row_info = f"Row {row_num}: " if row_num is not None else ""
        print(f"  [HTML] {row_info}Removed HTML tags: {original[:50]}... -> {cleaned[:50]}...")  # Comment this line to disable HTML logging

def log_pattern_removal(line, row_num=None, reason="pattern"):
    """Log pattern/short line removal - comment out the print line to disable pattern logs"""
    row_info = f"Row {row_num}: " if row_num is not None else ""
    reason_text = "short line" if reason == "short" else "pattern"
    print(f"  [PATTERN] {row_info}Removed line ({reason_text}): {line[:80]}...")  # Comment this line to disable pattern logging

def log_info(message):
    """Log info messages - comment out the print line to disable info logs"""
    print(message)  # Comment this line to disable info logging


### HTML Cleaning Functions ###
def clean_html_line(line, row_num=None):
    """
    Remove HTML tags from a single line of text.
    
    Parameters:
    -----------
    line : str
        Single line of text that may contain HTML tags
    row_num : int, optional
        Row number for logging
        
    Returns:
    --------
    str
        Line with HTML tags removed
    """
    original = line
    soup = BeautifulSoup(line, 'html.parser')    
    cleaned_text = soup.get_text(separator=' ', strip=True)    
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = cleaned_text.strip()
    
    log_html_removal(original, cleaned_text, row_num)
    
    return cleaned_text

def clean_html(text, row_num=None):
    """
    Remove HTML tags from a text string while preserving newlines.
    
    Parameters:
    -----------
    text : str
        Text string that may contain HTML tags
    row_num : int, optional
        Row number for logging
        
    Returns:
    --------
    str
        Text with HTML tags removed, newlines preserved
    """
    if pd.isna(text) or text is None:
        return text
    text = str(text)
    
    lines = text.split('\n')
    html_cleaned_lines = [clean_html_line(line, row_num) if line.strip() else '' for line in lines]
    return '\n'.join(html_cleaned_lines)

def remove_html_tags(df, column_name='content_text'):
    """
    Remove HTML tags from a specified column in a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data
    column_name : str, default='content_text'
        The name of the column containing HTML content
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with HTML tags removed from the specified column
    """
    df_cleaned = df.copy()
    
    if column_name in df_cleaned.columns:
        log_info(f"\n[INFO] Cleaning HTML tags from '{column_name}' column...")
        log_info(f"[INFO] Processing {len(df_cleaned)} rows...")
        df_cleaned[column_name] = df_cleaned.apply(lambda row: clean_html(row[column_name], row.name), axis=1)
        log_info(f"[INFO] HTML cleaning complete for {df_cleaned[column_name].notna().sum()} rows")
    else:
        log_info(f"[Warning] Column '{column_name}' not found in DataFrame")
        log_info(f"[INFO] Available columns: {df_cleaned.columns.tolist()}")
    
    return df_cleaned


### Boilerplate Patterns Removal Functions ###
def remove_common_patterns(text, row_num=None):
    """
    Remove lines containing common boilerplate patterns from web scraped content.
    
    Parameters:
    -----------
    text : str
        Text string that may contain boilerplate patterns
    row_num : int, optional
        Row number for logging
        
    Returns:
    --------
    str
        Text with lines containing common patterns removed, in lowercase
    """
    if pd.isna(text) or text is None:
        return text
    
    text = str(text)
    
    # Phone number patterns
    phone_patterns = [
        r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        r'\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
        r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}',
        r'Tel[:\s]+[\d\s\-\(\)\.]+',
        r'Phone[:\s]+[\d\s\-\(\)\.]+',
        r'Call[:\s]+[\d\s\-\(\)\.]+',
    ]
    
    # Email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    # URLs and links 
    url_patterns = [
        r'https?://[^\s]+',
        r'www\.[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,}(?:/[^\s]*)?',
        r'\b[a-zA-Z0-9][a-zA-Z0-9-]{2,}\.(?:com|org|net|edu|gov|uk|us)(?:/[^\s]*)?\b',
        r'\b[a-zA-Z0-9][a-zA-Z0-9-]{2,}\.(?:io|co\.uk)(?:/[^\s]*)?\b',
    ]
    
    # Physical addresses
    address_patterns = [
        r'\d+\s+[A-Za-z0-9\s,]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Lane|Ln|Way|Court|Ct|Place|Pl)[\s,]+[A-Za-z\s,]+(?:,\s*)?[A-Z]{2}\s+\d{5}(?:-\d{4})?',
        r'\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b',
        r'\b\d+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Lane|Ln|Way|Court|Ct|Place|Pl)\b',
    ]
    
    # Social media patterns
    social_patterns = [
        r'@[A-Za-z0-9_]+',
        r'facebook\.com/[^\s]+',
        r'twitter\.com/[^\s]+',
        r'instagram\.com/[^\s]+',
        r'linkedin\.com/[^\s]+',
        r'youtube\.com/[^\s]+',
    ]
    
    # Common navigation/footer text
    common_text_patterns = [
        r'©\s*\d{4}[\s\w]+',
        r'All rights reserved',
        r'Privacy Policy',
        r'Terms of Service',
        r'Terms and Conditions',
        r'Cookie Policy',
        r'Accept Cookies',
        r'We use cookies',
        r'Cookie Consent',
    ]
    
    # Combine all patterns
    all_patterns = phone_patterns + [email_pattern] + url_patterns + address_patterns + social_patterns + common_text_patterns
    
    # Add pattern for lines with repeated characters
    all_patterns.append(r'(.)\1{4,}')
    
    # Split text into lines first
    lines = text.split('\n')
    
    # For paragraph content, also split by sentences (periods, exclamation, question marks)
    all_segments = []
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        if len(line_stripped) > 200:
            sentences = re.split(r'([.!?]+\s+)', line_stripped)
            segments = []
            for i in range(0, len(sentences) - 1, 2):
                if i + 1 < len(sentences):
                    segments.append(sentences[i] + sentences[i + 1])
                else:
                    segments.append(sentences[i])
            if len(sentences) % 2 == 1:
                segments.append(sentences[-1])
            all_segments.extend([s.strip() for s in segments if s.strip()])
        else:
            all_segments.append(line_stripped)
    
    filtered_lines = []
    
    for segment in all_segments:
        segment_stripped = segment.strip()
        
        if not segment_stripped:
            continue
        
        if len(segment_stripped) < 30:
            log_pattern_removal(segment_stripped, row_num, reason="short")
            continue
        
        pattern_found = False
        matched_pattern = None
        for pattern in all_patterns:
            if re.search(pattern, segment_stripped, flags=re.IGNORECASE):
                pattern_found = True
                matched_pattern = pattern
                break
        
        if pattern_found:
            segment_len = len(segment_stripped)
            should_remove = False
            
            if segment_len < 200:
                should_remove = True
            else:
                if matched_pattern:
                    match = re.search(matched_pattern, segment_stripped, flags=re.IGNORECASE)
                    if match:
                        pattern_len = len(match.group())
                        if pattern_len > segment_len * 0.3:
                            should_remove = True
            
            if should_remove:
                log_pattern_removal(segment_stripped, row_num)
            else:
                filtered_lines.append(segment_stripped)
        else:
            filtered_lines.append(segment_stripped)
    
    result = '\n'.join(filtered_lines)
    return result.lower().strip()


def remove_boilerplate_patterns(df, column_name='content_text'):
    """
    Remove common boilerplate patterns from a specified column in a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data
    column_name : str, default='content_text'
        The name of the column containing text content
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with boilerplate patterns removed from the specified column
    """
    df_cleaned = df.copy()
    
    if column_name in df_cleaned.columns:
        log_info(f"[INFO] Removing boilerplate patterns from '{column_name}' column...")
        df_cleaned[column_name] = df_cleaned.apply(lambda row: remove_common_patterns(row[column_name], row.name), axis=1)
        log_info(f"[INFO] Boilerplate patterns removed from {df_cleaned[column_name].notna().sum()} rows")
    else:
        log_info(f"[Warning] Column '{column_name}' not found in DataFrame")
        log_info(f"[INFO] Available columns: {df_cleaned.columns.tolist()}")
    
    return df_cleaned
