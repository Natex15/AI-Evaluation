# step activation function
def sgn(value):
    if value > 0:
        return 1
    else:
        return -1


class Perceptron:
    # initialize perceptron hyperparameters
    def __init__(self, learning_rate=0.01, max_epochs=100, initial_bias=0.0):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.initial_bias = initial_bias
        self.weights = []
        self.biases = []
        self.classes = []

    # train one-vs-rest perceptron models on training data
    def fit(self, train_data):
        # collect unique classes from training data
        unique_classes = []
        for i in range(len(train_data)):
            label = train_data[i][-1]
            if label not in unique_classes:
                unique_classes.append(label)
        unique_classes.sort()
        self.classes = unique_classes

        num_features = len(train_data[0]) - 1

        # initialize weights and biases for each class using i and j
        self.weights = []
        self.biases = []
        for i in range(len(self.classes)):
            row = []
            for j in range(num_features):
                row.append(0.0)
            self.weights.append(row)
            self.biases.append(self.initial_bias)

        # train one perceptron per class using one-vs-rest
        for i in range(len(self.classes)):
            target_class = self.classes[i]

            # run training epochs for current class
            for epoch in range(self.max_epochs):
                weight_updated = False

                # iterate over all training samples
                for row_idx in range(len(train_data)):
                    sample = train_data[row_idx]
                    features = sample[:-1]
                    label = sample[-1]

                    # assign target value 1 for current class and -1 for others
                    if label == target_class:
                        target_value = 1
                    else:
                        target_value = -1

                    # calculate linear combination score
                    linear_combination = self.biases[i]
                    for j in range(num_features):
                        linear_combination += self.weights[i][j] * features[j]

                    predicted_value = sgn(linear_combination)

                    # update weights and bias when prediction is incorrect
                    if predicted_value != target_value:
                        difference = target_value - predicted_value
                        learning_step = self.learning_rate * difference

                        for j in range(num_features):
                            self.weights[i][j] += learning_step * features[j]
                        self.biases[i] += learning_step
                        weight_updated = True

                # stop early if no weights were updated in epoch
                if not weight_updated:
                    break

    # predict class with highest linear score
    def predict_one(self, features):
        best_score = -float("inf")
        best_class = self.classes[0]

        # calculate score for each class
        for i in range(len(self.classes)):
            linear_combination = self.biases[i]
            for j in range(len(features)):
                linear_combination += self.weights[i][j] * features[j]

            # update best class if higher score is found
            if linear_combination > best_score:
                best_score = linear_combination
                best_class = self.classes[i]

        return best_class

    # predict labels for all test samples
    def predict(self, test_data):
        predictions = []
        for i in range(len(test_data)):
            sample_row = test_data[i]
            num_features = len(self.weights[0])
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
