
from pathlib import Path
import pandas as pd


# Locate the project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Locate the dataset
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"


# Load the dataset
df = pd.read_csv(DATASET_PATH)


# Display basic information
print("\n--- DATASET OVERVIEW ---")
print("Total rows and columns:", df.shape)

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- FIRST 5 TRANSACTIONS ---")
print(df.head())


# Display dataset information
print("\n--- DATA TYPES AND NON-NULL VALUES ---")
df.info()


# Check fraud distribution
print("\n--- FRAUD DISTRIBUTION ---")
print(df["Class"].value_counts().sort_index())

print("\n--- FRAUD PERCENTAGE ---")
print(df["Class"].value_counts(normalize=True).sort_index() * 100)


# Check data quality
print("\n--- MISSING VALUES ---")
print("Total missing values:", df.isnull().sum().sum())

print("\n--- DUPLICATE RECORDS ---")
print("Total duplicates:", df.duplicated().sum())

# Investigate duplicate records

duplicate_rows = df[df.duplicated(keep=False)]

print("\n--- ALL DUPLICATE ROWS ---")
print("Total rows involved:", len(duplicate_rows))

print("\n--- DUPLICATE FRAUD DISTRIBUTION ---")
print(duplicate_rows["Class"].value_counts().sort_index())

print("\n--- SAMPLE DUPLICATE ROWS ---")
print(duplicate_rows.head(10))


# Statistical analysis of transactions

print("\n--- STATISTICAL SUMMARY ---")
print(df[["Time", "Amount"]].describe())


# Compare transaction amounts by class

print("\n--- AMOUNT ANALYSIS BY CLASS ---")

amount_analysis = df.groupby("Class")["Amount"].agg(
    ["count", "mean", "median", "min", "max"]
)

print(amount_analysis)
# Analyse transaction amount ranges

print("\n--- TRANSACTION AMOUNT RANGES ---")

print(
    pd.cut(
        df["Amount"],
        bins=[-1, 10, 50, 100, 500, 1000, float("inf")]
    ).value_counts().sort_index()
)

# Compare fraud distribution across amount ranges

amount_bins = pd.cut(
    df["Amount"],
    bins=[-1, 10, 50, 100, 500, 1000, float("inf")]
)

print("\n--- FRAUD DISTRIBUTION BY AMOUNT RANGE ---")

fraud_analysis = pd.crosstab(
    amount_bins,
    df["Class"],
    normalize="index"
) * 100

print(fraud_analysis)

# Count fraudulent transactions by amount range

print("\n--- FRAUD COUNTS BY AMOUNT RANGE ---")

fraud_counts = pd.crosstab(
    amount_bins,
    df["Class"]
)

print(fraud_counts)
print("\nTotal fraud transactions:", fraud_counts[1].sum())

# Analyse anonymised transaction features

print("\n--- FEATURE STATISTICAL SUMMARY ---")

feature_columns = [
    column for column in df.columns
    if column.startswith("V")
]

print(df[feature_columns].describe())

# Compare feature averages between legitimate and fraudulent transactions

print("\n--- FEATURE MEANS BY TRANSACTION CLASS ---")

feature_comparison = df.groupby("Class")[feature_columns].mean().T

print(feature_comparison)


# Rank features by difference between class averages

print("\n--- FEATURE DIFFERENCE RANKING ---")

feature_difference = (
    feature_comparison[1] - feature_comparison[0]
).abs()

feature_ranking = feature_difference.sort_values(
    ascending=False
)

print(feature_ranking.head(10))

