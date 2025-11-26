import pandas as pd
import os
from remove_boilerplates import remove_html_tags, remove_boilerplate_patterns
from lexical_quality_check import filter_by_ttr, filter_by_entropy, filter_by_mtld, filter_by_hapax_ratio, filter_by_repetition_ratio

data_path = os.path.join('data', 'Final_table_results.xlsx')

print("Loading Excel file...")
df_full = pd.read_excel(data_path)

print(f"Total rows in dataset: {len(df_full)}")

df_sample = df_full.sample(n=4000, random_state=10)

print(f"Sampled {len(df_sample)} rows")

# df_cleaned = remove_html_tags(df_sample, column_name='content_text')
# df_cleaned = remove_boilerplate_patterns(df_cleaned, column_name='content_text')
# df_cleaned = filter_by_ttr(df_cleaned, column_name='content_text')
# df_cleaned = filter_by_entropy(df_cleaned, column_name='content_text', entropy_threshold=4.0, drop_below=True)
# df_cleaned = filter_by_mtld(df_cleaned, column_name='content_text', mtld_threshold=9.0, ttr_threshold=0.90, min_words=5)
# df_cleaned = filter_by_hapax_ratio(df_cleaned, column_name='content_text', hapax_threshold=0.3, drop_below=True)
df_cleaned = filter_by_repetition_ratio(df_cleaned, column_name='content_text', repetition_threshold=0.3, drop_above=True)

print(f"\nData preprocessing complete!")
print(f"Final DataFrame shape: {df_cleaned.shape}")
