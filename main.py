import csv
import preprocess
import split
from gaussian_nb import NB_Classifier
from perceptron import Perceptron

#[LONGITUDE,LATITUDE,DEATHS_A,DEATHS_B,DEATHS_CIVILIANS,BEST,TYPE_OF_VIOLENCE,DURATION_DAYS,SIDE_B]
processed_data = preprocess.preprocess('conflict_data_phl.csv')

train, test = split.split_dataset(processed_data)

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

# Linear Perceptron (CPP vs Mindanao Groups)
train = preprocess.classify_groups(train)
test = preprocess.classify_groups(test)

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
