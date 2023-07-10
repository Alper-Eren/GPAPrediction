import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Veri kümesini yükleme
data = pd.read_csv('dataset.csv', delimiter=';')

# Veri ön işleme adımları
data['Gender'] = data['Gender'].map({'Female': 0, 'Male': 1})
data['Class'] = data['Class'].map({'1': 0, '2': 1, '3': 2, '4+': 3, 'Graduated': 4})
data['ParentEducation'] = data['ParentEducation'].map({'Primary': 0, 'High': 1, 'Degree': 2, 'Postgraduate': 3})
data['Income'] = data['Income'].map({'Low': 0, 'Moderate': 1, 'High': 2})
data['Continuity'] = data['Continuity'].map({'Low': 0, 'Moderate': 1, 'High': 2})
data['WeeklyWork'] = data['WeeklyWork'].map({'LessTen': 0, '11-20Hours': 1, '21-30Hours': 2, 'MoreThirty': 3})
data['Interest'] = data['Interest'].map({'Low': 0, 'Moderate': 1, 'High': 2})
data['ExamPreparation'] = data['ExamPreparation'].map({'Low': 0, 'Moderate': 1, 'High': 2})
data['Department'] = data['Department'].map({'Engineering': 0, 'Paramedic': 1, 'Other': 2})

data['GPA'] = data['GPA'].str.replace(',', '.').astype(float)
from sklearn.ensemble import RandomForestRegressor

# Extract the feature matrix X and the target variable y
X = data.drop('GPA', axis=1)  # Assuming 'GPA' is the target variable column
y = data['GPA']

# Create a Random Forest Regressor model
rf_model = RandomForestRegressor()

# Train the model on the entire dataset
rf_model.fit(X, y)

# Get the feature importances from the trained model
feature_importances = rf_model.feature_importances_

# Create a dictionary mapping features to their importances
importance_dict = dict(zip(X.columns, feature_importances))

# Sort the features based on their importances (in descending order)
sorted_importances = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

# Print the feature importances
for feature, importance in sorted_importances:
    print(f"{feature}: {importance}")

# Assuming your dataset is stored in a pandas DataFrame called 'data'
average_gpa = data['GPA'].mean()
print("Average GPA:", average_gpa)


