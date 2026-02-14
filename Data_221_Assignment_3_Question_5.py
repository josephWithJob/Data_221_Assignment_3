# Assignment 3 Question 1
# Joseph Krosel

import pandas
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Gets the csv
data_from_kidney_diseases_file = pandas.read_csv('kidney_disease.csv')

# Cleans up csv
data_from_kidney_diseases_file = data_from_kidney_diseases_file.fillna(0)

# Separates the data
classification_data_from_kidney_diseases_file = data_from_kidney_diseases_file['classification']
# classification_data_from_kidney_diseases_file = data_from_kidney_diseases_file['classification'].to_numpy()
data_from_kidney_diseases_file = data_from_kidney_diseases_file.drop('classification', axis=1)

# Since null data, we convert categorical data into 1s and 0s
# And with turning null data into 0s, many outliers will be made, skewing the data
data_from_kidney_diseases_file = data_from_kidney_diseases_file.replace({"normal":1, "abnormal":0, "present":1, "notpresent":0, "good":1, "poor":0, "yes":1, "no":0, "?":0, "\t?":0, "\tno":0, ' yes':1, '\tyes':1})
classification_data_from_kidney_diseases_file = classification_data_from_kidney_diseases_file.replace({"ckd\t":"ckd"})
classification_data_from_kidney_diseases_file = classification_data_from_kidney_diseases_file.to_numpy()

# stuff
features_train, features_test, labels_train, labels_test =(
    train_test_split(data_from_kidney_diseases_file,
                     classification_data_from_kidney_diseases_file,
                     test_size=0.33))

accuracy_test_list = []
for num_neighbors in [1,3,5,7,9]:
    knn_model = KNeighborsClassifier(n_neighbors=num_neighbors) # Define model

    # Trains data
    trained_knn_model = knn_model.fit(features_train, labels_train)

    # predicting
    predicted_labels = trained_knn_model.predict(features_test)

    # Using test to calculate how good it is
    accuracy_test_list.append(accuracy_score(labels_test, predicted_labels))

# Making the table for output
table_data_for_output = {
    'k':[1,2,3,4,5],
    'accuracy':accuracy_test_list}
print(pandas.DataFrame(table_data_for_output))


# Question 1: How changing k affects the behavior of the model
# Answer: As the k value increased, the accuracy decreased.
# Overall, reducing the effectiveness of the model.

# Question 2: Why very small values of k may cause overfitting
# Answer: The model becomes too sensitive to the because it doesn't have
# much data to read off of.

# Question 3: Why very large values of k may cause underfitting
# Answer: Very large values of k would bring issues of underfitting
# because it takes in too much data till the point that it can't recognise
# data trends. This inhibited its ability to predict data, causing high bias and
# low variance.