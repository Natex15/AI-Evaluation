import csv
import preprocess
import split
from gaussian_nb import NB_Classifier
from perceptron import Perceptron
from knn import KNN
import macrof1
import recall_precision

#[LONGITUDE,LATITUDE,DEATHS_A,DEATHS_B,DEATHS_CIVILIANS,BEST,TYPE_OF_VIOLENCE,DURATION_DAYS,SIDE_B]
processed_data = preprocess.preprocess('conflict_data_phl.csv')

train, test = split.split_dataset(processed_data)

# Classify groups into binary: CPP (0) vs Mindanao Groups (1)
train = preprocess.classify_groups(train)
test = preprocess.classify_groups(test)

# Gaussian Naive Bayes
nb_model = NB_Classifier()
nb_model.fit(train)
nb_predictions = nb_model.predict(test)

with open('nb_predictions.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['actual', 'predicted'])
    for sample_index in range(len(test)):
        actual_label = test[sample_index][-1]
        predicted_label = nb_predictions[sample_index]
        writer.writerow([actual_label, predicted_label])

# Linear Perceptron
perceptron_model = Perceptron()
perceptron_model.fit(train)
perceptron_predictions = perceptron_model.predict(test)

with open('perceptron_predictions.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['actual', 'predicted'])
    for sample_index in range(len(test)):
        actual_label = test[sample_index][-1]
        predicted_label = perceptron_predictions[sample_index]
        writer.writerow([actual_label, predicted_label])

# K-Nearest Neighbors
knn_model = KNN(k = 5, p = 2)
knn_model.fit(train)
knn_predictions = knn_model.predict(test)

with open('knn_predictions.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['actual', 'predicted'])
    for sample_index in range(len(test)):
        actual_label = test[sample_index][-1]
        predicted_label = knn_predictions[sample_index]
        writer.writerow([actual_label, predicted_label])

# Get target of test
test_actual = preprocess.get_target(test)

# Results
total_rows = len(processed_data)
total_cols = len(processed_data[0]) if total_rows > 0 else 0
num_features = total_cols - 1

nb_correct = recall_precision.count_correct(test_actual, nb_predictions)
perceptron_correct = recall_precision.count_correct(test_actual, perceptron_predictions)
knn_correct = recall_precision.count_correct(test_actual, knn_predictions)

print("Dataset Details")
print(f"Dataset Dimensions: {total_rows} rows x {total_cols} columns ({num_features} features, 1 target)")
print(f"Training Set Size: {len(train)}")
print(f"Testing Set Size (Number of Tests): {len(test)}\n")

print("Correct Predictions (Correct / Total Tests):")
print(f"Gaussian Naive Bayes: {nb_correct} / {len(test)} ({(nb_correct / len(test)) * 100:.2f}%)")
print(f"Perceptron: {perceptron_correct} / {len(test)} ({(perceptron_correct / len(test)) * 100:.2f}%)")
print(f"K-Nearest Neighbors: {knn_correct} / {len(test)} ({(knn_correct / len(test)) * 100:.2f}%)\n")

print("Evaluation Metrics:\n")

print("Recall and Precisions (Macro-Averaged):") #Macro averaged just means averaged per class
print(f"Gaussian Naive Bayes - Recall: {recall_precision.recall(test_actual, nb_predictions)}, Precision: {recall_precision.precision(test_actual, nb_predictions)}")
print(f"Perceptron           - Recall: {recall_precision.recall(test_actual, perceptron_predictions)}, Precision: {recall_precision.precision(test_actual, perceptron_predictions)}")
print(f"K-Nearest Neighbors  - Recall: {recall_precision.recall(test_actual, knn_predictions)}, Precision: {recall_precision.precision(test_actual, knn_predictions)}\n")

print("Macro F1 Scores:")
print("Gaussian Naive Bayes: " + str(macrof1.macrof1(test_actual, nb_predictions)))
print("Perceptron: " + str(macrof1.macrof1(test_actual, perceptron_predictions)))
print("K-Nearest Neighbors: " + str(macrof1.macrof1(test_actual, knn_predictions)))