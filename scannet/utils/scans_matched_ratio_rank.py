import os
import pandas as pd

parent = "/media/cc/Expansion/scannet/processed/scans"
# parent = "/media/cc/Expansion/scannet/processed/openset_scans/gpt4/openset_scans"
records = []   # store (folder, line_number, ratio)

for root, dirs, files in os.walk(parent):
    if 'instance_name_map.csv' in files and 'matched_instance_correspondence_to_00.csv' in files:
        folder = os.path.basename(root)
        inst_file = os.path.join(root, 'instance_name_map.csv')
        match_file = os.path.join(root, 'matched_instance_correspondence_to_00.csv')

        inst_lines = sum(1 for _ in open(inst_file))
        match_lines = sum(1 for _ in open(match_file))

        ratio = (match_lines - 1) / (inst_lines - 1 + 1e-6)
        records.append((folder, inst_lines, ratio))

# make a DataFrame
df = pd.DataFrame(records, columns=['folder', 'line_number', 'ratio'])

# ensure ratio is float
df["ratio"] = pd.to_numeric(df["ratio"], errors="coerce")

# drop anything that became NaN (non-numeric)
df = df.dropna(subset=["ratio"])

# Top 50and bottom 50 ratios
top50 = df.nlargest(50, 'ratio')
low50 = df.nsmallest(50, 'ratio')

print("Top 50:\n", top50[['folder', 'line_number', 'ratio']])
print("\nLowest 50:\n", low50[['folder', 'line_number', 'ratio']])
