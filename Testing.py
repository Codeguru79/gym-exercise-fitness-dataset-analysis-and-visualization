import pandas as pd
from scipy.stats import ttest_ind

df = pd.read_csv('gym_dataset_1000_rows.csv')

# T-Test 1: Gender vs Calories Burned (Independent t-test)
# Hypothesis
# H₀: Mean calories burned is same for males and females
# H₁: Mean calories burned is different

male = df[df['Gender'] == 'Male']['Calories_Burned']
female = df[df['Gender'] == 'Female']['Calories_Burned']

t_stat1, p_val1 = ttest_ind(male, female)

print("\nT-TEST 1: Gender vs Calories Burned\n")
print("T-statistic:", t_stat1)
print("P-value:", p_val1)

if p_val1 < 0.05:
    print("\nConclusion: Reject Null Hypothesis (Significant Difference)")
else:
    print("\nConclusion: Fail to Reject Null Hypothesis (No Significant Difference)")

print("\nAverage Calories Burned by Gender:")
print(df.groupby('Gender')['Calories_Burned'].mean())

print("-----------------------------------------------------------------------------------")


print("\nT-TEST 2: Gender comparison within 1-hour sessions\n")
df_1hr = df[(df['Session_Duration (hours)'] >= 0.9) & (df['Session_Duration (hours)'] <= 1.1)]

male = df_1hr[df_1hr['Gender'] == 'Male']['Calories_Burned']
female = df_1hr[df_1hr['Gender'] == 'Female']['Calories_Burned']

t_stat2, p_val2 = ttest_ind(male, female)

print("T-statistic:", t_stat2)
print("P-value:", p_val2)

if p_val2 < 0.05:
    print("\nConclusion: Reject H0 (Calories differ between genders)")
else:
    print("\nConclusion: Fail to Reject H0 (Calories are similar between genders)")


# Mean comparison
print("\nAverage Calories Burned:")
print(df_1hr.groupby('Gender')['Calories_Burned'].mean())

print("-----------------------------------------------------------------------------------")


print("\nT-TEST 3: Gender vs Average Heart Rate (Avg_BPM)\n")
# Hypothesis
# H₀: Mean heart rate is same for males and females
# H₁: Mean heart rate is different for  males and females

male = df[df['Gender'] == 'Male']['Avg_BPM']
female = df[df['Gender'] == 'Female']['Avg_BPM']

t_stat, p_val = ttest_ind(male, female)

print("T-statistic:", t_stat)
print("P-value:", p_val)

if p_val < 0.05:
    print("\nConclusion: Reject H0 (Heart rate differs between genders)")
else:
    print("\nConclusion: Fail to Reject H0 (Heart rate is similar between genders)")


# Mean values for better understanding
print("\nAverage Heart Rate by Gender:")
print(df.groupby('Gender')['Avg_BPM'].mean())