# GPA-Prediction
Guide for Predicting Values in the Dataset (GPA Prediction)

This code applies a machine learning algorithm to predict students' grade point averages (GPA) using a dataset named "dataset.csv". Below are the steps and information about how the code works:

## Loading the Dataset:

First, import the necessary libraries, including pandas and matplotlib.
Read the file "dataset.csv", which is a delimited file with semicolons (;).
Load the dataset into a DataFrame.

## Data Preprocessing:

Some columns in the dataset need to be converted to numerical values. These conversions are done using the "map()" function.
Categorical values in the 'Gender', 'Class', 'ParentEducation', 'Income', 'Continuity', 'WeeklyWork', 'Interest', 'ExamPreparation', and 'Department' columns are mapped to numerical values.
The 'GPA' column is converted to floating-point numbers, replacing commas with periods.

## Splitting the Data into Training and Testing Sets:

Use the "train_test_split()" function to split the dataset into training and testing sets.
The 'GPA' column is used as the target variable to be predicted.

## GPA Prediction with a Custom Algorithm:

Define a custom function called 'gpa_prediction_algorithm()'. This function makes predictions for each test example.
To make predictions, weights are assigned to each feature, and a weighted average is calculated.
The relationships between features and weights are stored in a dictionary called 'weights'.
The predicted GPA values are appended to a list and returned at the end.

## Evaluating the Predictions:

Calculate the absolute difference between the actual and predicted values to assess the accuracy of the predictions.
Calculate the percentage of successful predictions.
Plot a graph showing the distribution of actual and predicted values.

# Guide for Predicting Values in the Dataset (GPA Prediction)
This code uses machine learning to predict a student's grade point average (GPA). It requires a dataset to work with. The dataset should contain information about students, such as gender, class, parent's education, income, continuity, weekly work hours, interest level, exam preparation, department, and GPA.

## Requirements
To run this code, you need the following requirements:

- Python 3.x
- pandas
- matplotlib
- scikit-learn
## Installation
1.Install the requirements using Anaconda or pip:
```
pip install pandas matplotlib scikit-learn
```
2.Create a dataset named 'dataset.csv' and add columns with the relevant data. The columns should be as follows:

- 'Gender': Student's gender (Female/Male)
- 'Class': Student's class (1, 2, 3, 4+, Graduated)
- 'ParentEducation': Parent's education level (Primary, High, Degree, Postgraduate)
- 'Income': Family income level (Low, Moderate, High)
- 'Continuity': Student's continuity level (Low, Moderate, High)
- 'WeeklyWork': Weekly work hours (LessTen, 11-20Hours, 21-30Hours, MoreThirty)
- 'Interest': Student's interest level (Low, Moderate, High)
- 'ExamPreparation': Exam preparation level (Low, Moderate, High)
- 'Department': Student's department (Engineering, Paramedic, Other)
- 'GPA': Student's grade point average (decimal numbers, separated by periods)
3.Save your dataset to the 'dataset.csv' file.

4.Create a file named 'gpa_prediction_algorithm.py' and paste the above code into it.

5.Run the code and observe the results. The graph will show the relationship between the actual and predicted GPA values.

Using this code, you can experiment with different algorithms and models to predict students' grade point averages.

## Screenshots

![Figure_3](https://github.com/Alper-Eren/GPAPrediction/assets/100538269/7f445811-5267-4dc9-acc2-6a3a1c4ae93d)
![Figure_2](https://github.com/Alper-Eren/GPAPrediction/assets/100538269/df59c159-a8ba-4cc4-8203-68bcd3d179a7)
![Figure_1](https://github.com/Alper-Eren/GPAPrediction/assets/100538269/ed6c80f1-dc7b-4ae8-9586-566f08dd9599)


