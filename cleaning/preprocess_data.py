import pandas as pd
import os
from remove_boilerplates import remove_html_tags, remove_boilerplate_patterns

data_path = os.path.join('data', 'Final_table_results.xlsx')

print("Loading Excel file...")
df_full = pd.read_excel(data_path)

print(f"Total rows in dataset: {len(df_full)}")

# df_sample = df_full.sample(n=4000, random_state=12)
df_sample = df_full.sample(n=50, random_state=10)

print(f"Sampled {len(df_sample)} rows")

df_cleaned = remove_html_tags(df_sample, column_name='content_text')
df_cleaned = remove_boilerplate_patterns(df_cleaned, column_name='content_text')

print(f"\nData preprocessing complete!")
print(f"Final DataFrame shape: {df_cleaned.shape}")
