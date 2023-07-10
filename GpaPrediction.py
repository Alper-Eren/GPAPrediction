import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('dataset.csv', delimiter=';')

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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('GPA', axis=1), data['GPA'], test_size=0.2, random_state=None) 

# Custom algorithm for GPA prediction
def gpa_prediction_algorithm(X):
    weights = {
        'Gender': 3.97/100,
        'Class': 14.16/100,
        'ParentEducation': 13.33/100,
        'Income': 2.73/100,
        'Continuity': 18.24/100,
        'WeeklyWork': 9.14/100,
        'Interest': 14.24/100,
        'ExamPreparation': 10.47/100,
        'Department': 13.66/100,
    }

    gpa_predictions = []
    
    
    for _, row in X.iterrows():
        #Calculating minimum of dataset
        average_gpa = data['GPA'].mean() #2.97
        avg = average_gpa - (4 - average_gpa) 

        # Calculate the weighted average based on the features and their weights
        for feature, weight in weights.items():
            avg += row[feature] * weight

        gpa_predictions.append(avg)

    return gpa_predictions

# Make predictions using the custom algorithm
y_pred = gpa_prediction_algorithm(X_test)

# Calculate the absolute difference between actual and predicted values
diff = abs(y_test - y_pred)

# Calculate the percentage of successful predictions
percentage_success = (sum(diff < 0.5) / len(y_test)) * 100

# Plotting the actual values and predicted values
plt.scatter(y_test, y_pred)
plt.plot([0, 4], [0, 4], color='red', linestyle='--')  # x=y line
plt.xlabel('Actual GPA')
plt.ylabel('Predicted GPA')
plt.title('Actual vs. Predicted GPA\nPercentage of successful predictions: {:.2f}%'.format(percentage_success))
plt.show()