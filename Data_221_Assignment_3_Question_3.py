# Assignment 3 Question 1
# Joseph Krosel

import pandas
from sklearn.model_selection import train_test_split

# Gets the csv
data_from_kidney_diseases_file = pandas.read_csv('kidney_disease.csv')
classification_data_from_kidney_diseases_file = pandas.read_csv('kidney_disease.csv', usecols=['classification'])

data_from_kidney_diseases_file = data_from_kidney_diseases_file.drop('classification', axis=1)

features_train, features_test, labels_train, labels_test =(
    train_test_split(data_from_kidney_diseases_file,
                     classification_data_from_kidney_diseases_file,
                     test_size=0.3, random_state=30))
# MISSING STUFF BY THE WAY (Have to split the data)

# Question 1: Why we should not train and test a model on the same data
# Answer: We should not train and test on the same data because
# when you separate them, it is easy to find if your model is overfitted
# or not.

# Question 2: What the purpose of the testing set is
# Answer: The purpose of the testing set is the accurately test the
# model on new data.