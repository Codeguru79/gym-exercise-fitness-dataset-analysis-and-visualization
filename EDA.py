import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, pearsonr, f_oneway


df = pd.read_csv('gym_dataset_1000_rows.csv')

print(df.head())
print("-----------------------------------------------------------------------------------")
print(df.shape)
print("-----------------------------------------------------------------------------------")
print(df.columns)
print("-----------------------------------------------------------------------------------")
print(df.info())
print("-----------------------------------------------------------------------------------")

print(df.describe())
print("-----------------------------------------------------------------------------------")

print(df.isnull().sum())
print("-----------------------------------------------------------------------------------")

df_num = df.select_dtypes(include=['int64','float64'])

# This heatmap displays correlation between all numerical variables to identify strong relationships
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), fmt = '.1f',annot=True)
plt.title('Correlation Heatmap')
plt.show()

# This count plot shows how many people fall under each workout type
sns.countplot(x='Workout_Type', data=df)
plt.title('Count of Participants by Workout Type')
plt.xticks(rotation=30)
plt.show()

# This count plot shows the distribution of experience levels among participants
sns.countplot(x='Experience_Level', data=df)
plt.title('Count by Experience Level')
plt.show()

# This count plot shows how frequently participants work out per week
sns.countplot(x='Workout_Frequency (days/week)', data=df)
plt.title('Workout Frequency Distribution')
plt.show()


# This count plot shows the number of male and female participants in the dataset
sns.countplot(x='Gender', data=df)
plt.title('Count of Participants by Gender')
plt.show()


for col in df_num.columns:
    plt.figure(figsize=(10,6))
    sns.histplot(data=df, x=col, kde=True, bins=25,hue='Gender')
    plt.title(f'Distribution of {col}')
    plt.show()  

for col in df_num.columns:
    plt.figure(figsize=(10,6))
    
    # This boxplot compares the distribution of each numerical feature across gender to identify differences and outliers
    sns.boxplot(x='Gender', y=col, data=df, palette='viridis')
    
    plt.title(f'{col} Distribution by Gender')
    plt.xlabel('Gender')
    plt.ylabel(col)
    plt.show()

print("-----------------------------------------------------------------------------------")
print("Outlier Detection:")
for col in df_num.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    print(f"\nOutliers in {col}:")
    print(outliers[[col]])


# This scatter plot shows the relationship between workout duration and calories burned
sns.scatterplot(x='Session_Duration (hours)', y='Calories_Burned', data=df)
plt.title('Calories vs Session Duration')
plt.show()


# # This correlation test checks the strength and significance of the relationship between session duration and calories burned
corr, p_val = pearsonr(df['Session_Duration (hours)'], df['Calories_Burned'])
print("Correlation:", corr, p_val)


