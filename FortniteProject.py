import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import seaborn as sns

#create initial dataframe
main_df = pd.read_csv("Fortnite Statistics.csv")

#revise dataframe by removing not needed metrics
revised_df = main_df.drop(columns=['Mental State', 'Date', 'Time of Day'])

#make sure it worked
print(revised_df)

#convert accuracy to float from string
revised_df['Accuracy'] = revised_df['Accuracy'].str.replace('%', '').astype(float) / 100

#select features and target
X = revised_df[['Eliminations', 'Damage to Players', 'Accuracy']]
y = revised_df['Placed']

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#linear regression
model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("R²:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# Combine scaled features and predictions into one DataFrame for plotting
plot_df = X_test.copy()
plot_df['Placed (Actual)'] = y_test
plot_df['Placed (Predicted)'] = y_pred

# Plot relationships
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, feature in zip(axes, ['Eliminations', 'Damage to Players', 'Accuracy']):
    sns.regplot(x=plot_df[feature], y=plot_df['Placed (Actual)'], ax=ax, scatter_kws={'alpha':0.6})
    ax.set_title(f'{feature} vs Placement')
    ax.set_ylabel('Placed')
    ax.invert_yaxis()  # Lower placement = better performance
    ax.set_xlabel(feature)

plt.tight_layout()
plt.show()
