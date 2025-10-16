import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Path to dataset
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "heart.csv")

# Load the dataset
df = pd.read_csv(DATA_PATH)

print("✅ Dataset loaded successfully for EDA")
print("Shape of data:", df.shape)
print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Data Information ---")
print(df.info())

print("\n--- Summary Statistics ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isnull().sum())

# Visualizations
sns.set(style="whitegrid", palette="pastel")

# 1. Target variable distribution
plt.figure(figsize=(5, 4))
sns.countplot(x='target', data=df)
plt.title("Target Class Distribution (0 = No Disease, 1 = Disease)")
plt.show()

# 2. Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# 3. Age distribution by disease
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x='age', hue='target', multiple='stack', bins=20)
plt.title("Age Distribution by Disease Status")
plt.show()
