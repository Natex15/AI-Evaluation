def sgn(value):
    if value > 0:
        return 1
    else:
        return -1


class Perceptron:
    def __init__(self, learning_rate=0.01, max_epochs=100, initial_bias=0.0):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.bias = initial_bias
        self.weights = []
        self.classes = []

    def fit(self, train_data):
        unique_classes = []
        for sample_row in train_data:
            label = sample_row[-1]
            if label not in unique_classes:
                unique_classes.append(label)
        self.classes = unique_classes

        num_features = len(train_data[0]) - 1

        self.weights = []
        for feature_index in range(num_features):
            self.weights.append(0.0)

        for epoch in range(self.max_epochs):
            weight_updated = False

            for sample_row in train_data:
                features = sample_row[:-1]
                label = sample_row[-1]

                if label == self.classes[-1]:
                    target_value = 1
                else:
                    target_value = -1

                linear_combination = 0.0
                for feature_index in range(num_features):
                    linear_combination += self.weights[feature_index] * features[feature_index]
                linear_combination += self.bias

                predicted_value = sgn(linear_combination)

                if predicted_value != target_value:
                    difference = target_value - predicted_value
                    learning_step = self.learning_rate * difference

                    for feature_index in range(num_features):
                        self.weights[feature_index] += learning_step * features[feature_index]
                    self.bias += learning_step

                    weight_updated = True

            if not weight_updated:
                break

    def predict_one(self, features):
        linear_combination = 0.0
        for feature_index in range(len(features)):
            linear_combination += self.weights[feature_index] * features[feature_index]
        linear_combination += self.bias

        predicted_value = sgn(linear_combination)

        if predicted_value == 1:
            return self.classes[-1]
        else:
            return self.classes[0]

    def predict(self, test_data):
        predictions = []
        for sample_row in test_data:
            num_features = len(self.weights)
            if len(sample_row) > num_features:
                features = sample_row[:-1]
            else:
                features = sample_row

            predicted_label = self.predict_one(features)
            predictions.append(predicted_label)
        return predictions


LinearPerceptron = Perceptron

if __name__ == "__main__":
    import csv
    import preprocess
    import split

    data = preprocess.preprocess('conflict_data_phl.csv')
    train_data, test_data = split.split_dataset(data)
    train_data = preprocess.classify_groups(train_data)
    test_data = preprocess.classify_groups(test_data)

    model = Perceptron()
    model.fit(train_data)
    predictions = model.predict(test_data)

    with open('perceptron_predictions.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['actual', 'predicted'])
        for sample_index in range(len(test_data)):
            actual_label = test_data[sample_index][-1]
            predicted_label = predictions[sample_index]
            writer.writerow([actual_label, predicted_label])
