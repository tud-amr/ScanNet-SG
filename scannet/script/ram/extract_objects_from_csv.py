import pandas as pd
import re

csv_path = "/media/cc/Expansion/scannet/processed/object_name_counts.csv"

# read csv (your file has 2 columns: object_name, count)
df = pd.read_csv(csv_path)

def clean_name(s: str) -> str:
    s = str(s).strip()
    # remove leading numbers like: "12. chair" / "12) chair" / "12-chair"
    s = re.sub(r"^\d+\s*[\.\)\-_:]*\s*", "", s)
    # normalize spaces
    s = re.sub(r"\s+", " ", s)
    # title case
    return s.title()

names = [clean_name(x) for x in df["object_name"].tolist()]

# (optional) remove empty strings
# names = [x for x in names if x]

# output in required format
print("openimages_scannet_509 = [")
for n in names:
    print(f"    '{n}',")
print("]")
