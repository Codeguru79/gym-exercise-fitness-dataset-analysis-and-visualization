import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv('gym_dataset_1000_rows.csv')

print("Model 1: Session Duration → Calories Burned")
X1 = df[['Session_Duration (hours)']]
y1 = df['Calories_Burned']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

model1 = LinearRegression()
model1.fit(X1_train, y1_train)

y1_pred = model1.predict(X1_test)

print("Model 1 R2:", r2_score(y1_test, y1_pred))
print("Model 1 MSE:", mean_squared_error(y1_test, y1_pred))

# Model 1 Interpretation

# (Session Duration → Calories Burned)

# R² Score = 0.813
# Means 81.3% of variation in calories burned is explained by workout duration
# This is a strong model
# MSE = 13919.63
# This is the average squared error between actual and predicted values
# Since calories are large values, this MSE is acceptable
# What This Means (Simple Understanding)
# As session duration increases, calories burned also increase
# Model captures this relationship very well
# Final Conclusion (Write This in Report)

# The linear regression model shows a strong relationship between session duration and calories burned, with an R² value of 0.813. This indicates that workout duration is a significant predictor of calories burned. The model demonstrates good predictive performance with acceptable error levels.

# Hypothesis Decision
# H₀: No relationship
# H₁: Significant relationship

# Since model is strong →
# Reject H₀, Accept H₁

print("-----------------------------------------------------------------------------------")
print("Model 2: Avg BPM → Calories Burned")
X2 = df[['Avg_BPM']]
y2 = df['Calories_Burned']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

model2 = LinearRegression()
model2.fit(X2_train, y2_train)

y2_pred = model2.predict(X2_test)

print("Model 2 R2:", r2_score(y2_test, y2_pred))
print("Model 2 MSE:", mean_squared_error(y2_test, y2_pred))

# Model 2 Interpretation

# (Avg BPM → Calories Burned)

# R² Score = 0.136
# Only 13.6% of variation in calories burned is explained
# This is a weak model
# MSE = 64367.81
# Error is much higher than Model 1
# Indicates poor prediction accuracy
# What This Means (Important Insight)
# Heart rate alone does NOT strongly determine calories burned
# Other factors (like duration, weight, workout type) matter more

# This is actually a very good real-world finding

# Final Conclusion (Write This in Report)

# The regression model between average heart rate and calories burned shows a weak relationship, with an R² value of 0.136. This indicates that average heart rate alone is not a strong predictor of calories burned. The high mean squared error further confirms the model’s low predictive accuracy.

# Hypothesis Decision
# H₀: No relationship
# H₁: Significant relationship

# Since model is weak →
# Fail to strongly reject H₀
# (or say: weak evidence against H₀)
 

print("-----------------------------------------------------------------------------------")
print("Model 3: BMI → Fat Percentage")
X3 = df[['BMI']]
y3 = df['Fat_Percentage']

X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=42)

model3 = LinearRegression()
model3.fit(X3_train, y3_train)

y3_pred = model3.predict(X3_test)

print("Model 3 R2:", r2_score(y3_test, y3_pred))
print("Model 3 MSE:", mean_squared_error(y3_test, y3_pred))

# Model 3 Interpretation

# (BMI → Fat Percentage)

# R² Score = 0.003
# Only 0.3% of variation is explained
# This is an extremely weak / almost no relationship
# MSE = 38.25
# Error is relatively low compared to Model 2
# But since R² is almost zero → model is not meaningful
# What This Means (Very Important Insight)
# BMI does NOT strongly predict fat percentage in this dataset
# Even though in theory they should be related, here:
# Data variability is high
# Other hidden factors affect fat %

# This is actually a strong analytical conclusion

# Final Conclusion (Write This in Report)

# The regression analysis between BMI and fat percentage shows a very weak relationship, with an R² value close to zero. This indicates that BMI is not a reliable predictor of fat percentage in this dataset. Despite a relatively low error, the model fails to explain the variation in fat percentage.

# Hypothesis Decision
# H₀: No relationship
# H₁: Significant relationship

# Since R² ≈ 0 →
# Accept H₀ (no significant relationship)

print("-----------------------------------------------------------------------------------")
print("Visualizing Model Predictions")

plt.scatter(X1_test, y1_test)
plt.plot(X1_test, y1_pred)
plt.title('Model 1: Calories vs Session Duration')
plt.show()

plt.scatter(X2_test, y2_test)
plt.plot(X2_test, y2_pred)
plt.title('Model 2: Calories vs Avg BPM')
plt.show()

plt.scatter(X3_test, y3_test)
plt.plot(X3_test, y3_pred)
plt.title('Model 3: Fat % vs BMI')
plt.show()