
from pathlib import Path
import pandas as pd


# ==================================================
# 1. LOCATE PROJECT AND DATASET
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATASET_PATH = (
    PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
)

PROCESSED_DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "transactions.csv"
)


# ==================================================
# 2. LOAD ORIGINAL DATASET
# ==================================================

print("\nLoading original dataset...")

df = pd.read_csv(RAW_DATASET_PATH)


# ==================================================
# 3. BASIC DATASET INVESTIGATION
# ==================================================

print("\n--- ORIGINAL DATASET ---")

print("Rows and columns:", df.shape)

print("\n--- DUPLICATE INVESTIGATION ---")

print("Duplicate occurrences:", df.duplicated().sum())

print("\n--- MISSING VALUES ---")

print("Total missing values:", df.isnull().sum().sum())


# ==================================================
# 4. INVESTIGATE DUPLICATE RECORDS
# ==================================================

duplicate_rows = df[df.duplicated(keep=False)]

print("\n--- DUPLICATE RECORD ANALYSIS ---")

print(
    "Total rows involved in duplicate groups:",
    len(duplicate_rows)
)

print("\n--- DUPLICATE DISTRIBUTION BY CLASS ---")

print(
    duplicate_rows["Class"].value_counts().sort_index()
)

print("\n--- SAMPLE DUPLICATE RECORDS ---")

print(duplicate_rows.head(5))


# ==================================================
# 5. INVESTIGATE FRAUDULENT DUPLICATE RECORDS
# ==================================================

fraud_duplicate_rows = duplicate_rows[
    duplicate_rows["Class"] == 1
]

print("\n--- DUPLICATE FRAUD COUNT ---")

print(
    "Fraud rows in duplicate groups:",
    len(fraud_duplicate_rows)
)


# ==================================================
# 
# ==================================================
# 6. REMOVE EXACT DUPLICATE RECORDS
# ==================================================

print("\n--- DUPLICATE REMOVAL ---")

original_row_count = len(df)

df = df.drop_duplicates()

cleaned_row_count = len(df)

removed_rows = original_row_count - cleaned_row_count

print("Original rows:", original_row_count)
print("Rows after duplicate removal:", cleaned_row_count)
print("Duplicate rows removed:", removed_rows)


# ==================================================
# 7. SAVE CLEANED DATASET
# ==================================================

df.to_csv(
    PROCESSED_DATASET_PATH,
    index=False
)

print("\n--- CLEANED DATASET SAVED ---")

print("Saved location:", PROCESSED_DATASET_PATH)
print("Final dataset size:", df.shape)



# ==================================================
# 7. FINAL STATUS
# ==================================================

print("\n--- PREPARATION STATUS ---")

print("Original dataset remains unchanged.")

print("No rows have been deleted.")

print("No missing values were filled.")

print("No duplicate records were removed.")

print("\nInitial dataset preparation completed!")

# ==================================================
# 8. CHECK FOR CONFLICTING DUPLICATE LABELS
# ==================================================

feature_columns = df.columns.drop("Class")

duplicate_feature_mask = df.duplicated(
    subset=feature_columns,
    keep=False
)

duplicate_feature_rows = df[duplicate_feature_mask]

conflicting_label_groups = (
    duplicate_feature_rows
    .groupby(list(feature_columns))["Class"]
    .nunique()
)

conflicting_label_groups = conflicting_label_groups[
    conflicting_label_groups > 1
]

print("\n--- CONFLICTING DUPLICATE LABEL INVESTIGATION ---")

print(
    "Rows with repeated feature combinations:",
    len(duplicate_feature_rows)
)

print(
    "Feature groups with conflicting fraud labels:",
    len(conflicting_label_groups)
)


# ==================================================
# 8. SEPARATE FEATURES AND TARGET
# ==================================================

print("\n--- FEATURE AND TARGET SEPARATION ---")

# X contains all input features
X = df.drop(columns=["Class"])

# y contains the target labels
y = df["Class"]

print("Feature dataset shape:", X.shape)
print("Target dataset shape:", y.shape)

print("\n--- TARGET CLASS DISTRIBUTION ---")

print(y.value_counts().sort_index())

print("\n--- TARGET CLASS PERCENTAGES ---")

print(y.value_counts(normalize=True).sort_index() * 100)


# ==================================================
# 9. SPLIT DATA INTO TRAINING AND TESTING SETS
# ==================================================

from sklearn.model_selection import train_test_split

print("\n--- TRAIN TEST SPLIT ---")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)

print("\n--- TRAINING FRAUD DISTRIBUTION ---")
print(y_train.value_counts())

print("\n--- TESTING FRAUD DISTRIBUTION ---")
print(y_test.value_counts())


# ==================================================
# 10. SAVE TRAINING AND TESTING DATASETS
# ==================================================

print("\n--- SAVING TRAINING AND TESTING DATASETS ---")

# Combine features and target labels
train_data = X_train.copy()
train_data["Class"] = y_train

test_data = X_test.copy()
test_data["Class"] = y_test


# Save training data
train_data.to_csv(
    PROJECT_ROOT / "data" / "processed" / "train.csv",
    index=False
)

# Save testing data
test_data.to_csv(
    PROJECT_ROOT / "data" / "processed" / "test.csv",
    index=False
)

print("Training dataset saved:", train_data.shape)
print("Testing dataset saved:", test_data.shape)

print("\nTraining and testing datasets saved successfully!")





