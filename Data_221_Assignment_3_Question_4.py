# Assignment 3 Question 1
# Joseph Krosel

import pandas
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

knn_model = KNeighborsClassifier(n_neighbors=5) # Define model

# Trains data
trained_knn_model = knn_model.fit(features_train, labels_train)

# predicting
predicted_labels = trained_knn_model.predict(features_test)

# Confusion matrix
confusion_matrix_of_classification = confusion_matrix(labels_test, predicted_labels)
print(confusion_matrix_of_classification)

# Using test to calculate how good it is
accuracy_of_test = accuracy_score(labels_test, predicted_labels)
print(accuracy_of_test)
precision_of_test = precision_score(labels_test, predicted_labels, pos_label='ckd')
print(precision_of_test)
recall_of_test = recall_score(labels_test, predicted_labels, pos_label='ckd')
print(recall_of_test)
f1_of_test = f1_score(labels_test, predicted_labels, pos_label='ckd')
print(f1_of_test)

# Question 1: What True Positive, True Negative, False Positive, and
# False Negative mean in the context of kidney disease prediction
# Answer: In the context of this model, the true positive is the prediction
# guessing ckd and it being ckd. The true negative is the model predicting
# notckd and it being notckd. The false positive is the model prediction
# ckd and it being notckd. While the false negative is the prediction being
# notckd and it being ckd.

# Question 2: Why accuracy alone may not be enough to evaluate a classification model
# Answer: Accuracy alone is not good enough because models may just be predicting
# the most outcome that majority would have. Which is why we spread our options,
# which would give different outcomes, allowing us to catch these lacking qualities in
# the model

# Question 3: Which metric is most important if missing a kidney disease case is very serious, and why
# Answer:
