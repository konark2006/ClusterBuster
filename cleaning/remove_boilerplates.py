import pandas as pd
from bs4 import BeautifulSoup
import re


def clean_html(text):
    """
    Remove HTML tags from a text string.
    
    Parameters:
    -----------
    text : str
        Text string that may contain HTML tags
        
    Returns:
    --------
    str
        Text with HTML tags removed
    """
    if pd.isna(text) or text is None:
        return text
    text = str(text)
    
    soup = BeautifulSoup(text, 'html.parser')    
    cleaned_text = soup.get_text(separator=' ', strip=True)    
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    return cleaned_text.strip()

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
    
    # Apply cleaning function to the specified column
    if column_name in df_cleaned.columns:
        print(f"Cleaning HTML tags from '{column_name}' column...")
        df_cleaned[column_name] = df_cleaned[column_name].apply(clean_html)
        print(f"HTML tags removed from {df_cleaned[column_name].notna().sum()} rows")
    else:
        print(f"Warning: Column '{column_name}' not found in DataFrame")
        print(f "Available columns: {df_cleaned.columns.tolist()}")
    
    return df_cleaned

