import pandas as pd

# Load Yale OIR file
df = pd.read_csv("w026-first-year-geo-origin(in).csv")

# Inspect the columns to make sure we grab the right one
print(df.columns)

# Assuming it has a column called "State" (or similar) and one called "2024"
# Replace column names if they are slightly different in file
df_sorted = df[["State", "2024"]].copy()

# Sort by 2024 student counts descending
df_sorted = df_sorted.sort_values("2024", ascending=False).reset_index(drop=True)

# Add percentage of total students for extra clarity
df_sorted["percent_2024"] = df_sorted["2024"] / df_sorted["2024"].sum() * 100

# Save the result
df_sorted.to_csv("yale_oir_2024_sorted.csv", index=False)

print(df_sorted.head(52))  # Show top 20 states by student counts
