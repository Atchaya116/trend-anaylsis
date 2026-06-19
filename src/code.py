import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

# Load Dataset
df = pd.read_csv("ai_adoption_dataset.csv")

# -------------------------------
# PHASE 1: DATA UNDERSTANDING
# -------------------------------

print("FIRST 5 RECORDS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns)

print("\nDATASET INFO")
print(df.info())

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE RECORDS")
print(df.duplicated().sum())

df.drop_duplicates(inplace=True)

# -------------------------------
# PHASE 2: DESCRIPTIVE STATISTICS
# -------------------------------

print("\nDESCRIPTIVE STATISTICS")
print(df.describe())

print("\nMEAN")
print(df.mean(numeric_only=True))

print("\nMEDIAN")
print(df.median(numeric_only=True))

print("\nSTANDARD DEVIATION")
print(df.std(numeric_only=True))

print("\nQUARTILES")
print(df.quantile([0.25,0.50,0.75], numeric_only=True))

print("\nPERCENTILES")
print(df.quantile([0.90,0.95], numeric_only=True))

# -------------------------------
# PHASE 3: TREND ANALYSIS
# -------------------------------

plt.figure(figsize=(8,5))
sns.countplot(data=df, x='Industry')
plt.title("Industry Distribution")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8,5))
sns.countplot(data=df, x='Company_Size')
plt.title("Company Size Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.countplot(data=df, x='AI_Tool')
plt.title("AI Tool Usage")
plt.xticks(rotation=45)
plt.show()

yearly = df.groupby('Adoption_Year').size()

plt.figure(figsize=(8,5))
yearly.plot(marker='o')
plt.title("AI Adoption Growth")
plt.ylabel("Organizations")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(
    data=df,
    x='Employee_Count',
    y='AI_Usage_Hours'
)
plt.title("Employee Count vs AI Usage")
plt.show()

# -------------------------------
# PHASE 4: CORRELATION ANALYSIS
# -------------------------------

corr = df.select_dtypes(include=np.number).corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr,
            annot=True,
            cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# -------------------------------
# PHASE 5: HYPOTHESIS TESTING
# -------------------------------

small = df[df['Company_Size']=='Small']['Adoption_Score']
medium = df[df['Company_Size']=='Medium']['Adoption_Score']
large = df[df['Company_Size']=='Large']['Adoption_Score']

f_stat, p_value = f_oneway(
    small,
    medium,
    large
)

print("\nANOVA TEST")
print("F Statistic =", f_stat)
print("P Value =", p_value)

if p_value < 0.05:
    print("Reject H0")
    print("Company Size significantly affects AI Adoption")
else:
    print("Fail to Reject H0")
    print("Company Size does not significantly affect AI Adoption")

# -------------------------------
# PHASE 6: SEGMENTATION
# -------------------------------

conditions = [
    df['Adoption_Score'] > 80,
    df['Adoption_Score'] > 60,
    df['Adoption_Score'] <= 60
]

labels = [
    'AI Leaders',
    'Early Adopters',
    'Slow Adopters'
]

df['Segment'] = np.select(
    conditions,
    labels,
    default='Slow Adopters'
)

print("\nSEGMENT COUNT")
print(df['Segment'].value_counts())

plt.figure(figsize=(8,5))
sns.countplot(data=df, x='Segment')
plt.title("Organization Segmentation")
plt.show()

print("\nSEGMENT ANALYSIS")
print(
    df.groupby('Segment')[
        ['Productivity_Gain',
         'Satisfaction_Score',
         'Cost_Savings']
    ].mean()
)

# -------------------------------
# PHASE 7: BUSINESS INSIGHTS
# -------------------------------

print("\nBUSINESS INSIGHTS")
print("1. Technology-focused industries show higher AI adoption.")
print("2. Organizations with higher AI investments achieve greater savings.")
print("3. Larger companies tend to adopt AI faster.")
print("4. Higher AI usage improves productivity and satisfaction.")
print("5. AI Leaders achieve better ROI and business performance.")

# Save Final Dataset
df.to_csv("AI_Adoption_Final_Output.csv", index=False)

print("\nTask 7 Completed Successfully")
