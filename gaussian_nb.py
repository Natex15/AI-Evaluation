import math

class NB_Classifier:
    def __init__(self, smoothing=1e-9):
        self.smoothing = smoothing
        self.classes = []
        self.priors = {}
        self.means = {}
        self.variances = {}

    def fit(self, train_data):
        total_samples = len(train_data)

        grouped_by_class = {}
        for sample_row in train_data:
            features = sample_row[:-1]
            label = sample_row[-1]
            if label not in grouped_by_class:
                grouped_by_class[label] = []
            grouped_by_class[label].append(features)

        self.classes = list(grouped_by_class.keys())
        num_features = len(train_data[0]) - 1

        for label, class_samples in grouped_by_class.items():
            class_count = len(class_samples)
            self.priors[label] = class_count / total_samples
            class_means = []
            for feature_index in range(num_features):
                feature_total = 0.0
                for features in class_samples:
                    feature_total += features[feature_index]
                class_means.append(feature_total / class_count)
            self.means[label] = class_means

            class_variances = []
            for feature_index in range(num_features):
                squared_diff_sum = 0.0
                for features in class_samples:
                    squared_diff_sum += (features[feature_index] - class_means[feature_index]) ** 2
                class_variances.append((squared_diff_sum / class_count) + self.smoothing)
            self.variances[label] = class_variances

    def calculate_log_likelihood(self, feature_value, mean, variance):
        coefficient = -0.5 * math.log(2 * math.pi * variance)
        exponent = -((feature_value - mean) ** 2) / (2 * variance)
        return coefficient + exponent

    def predict_one(self, features):
        best_label = None
        highest_log_prob = -float("inf")

        for label in self.classes:
            log_prob = math.log(self.priors[label])
            for feature_index in range(len(features)):
                mean = self.means[label][feature_index]
                variance = self.variances[label][feature_index]
                log_prob += self.calculate_log_likelihood(features[feature_index], mean, variance)

            if log_prob > highest_log_prob:
                highest_log_prob = log_prob
                best_label = label

        return best_label

    def predict(self, test_data):
        predictions = []
        for sample_row in test_data:
            num_features = len(self.means[self.classes[0]])
            if len(sample_row) > num_features:
                features = sample_row[:-1]
            else:
                features = sample_row

            predicted_label = self.predict_one(features)
            predictions.append(predicted_label)
        return predictions


GaussianNB = NB_Classifier

if __name__ == "__main__":
    import csv
    import preprocess
    import split

    data = preprocess.preprocess('conflict_data_phl.csv')
    train_data, test_data = split.split_dataset(data)

    model = NB_Classifier()
    model.fit(train_data)
    predictions = model.predict(test_data)

    with open('nb_predictions.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['actual', 'predicted'])
        for sample_index in range(len(test_data)):
            actual_label = test_data[sample_index][-1]
            predicted_label = predictions[sample_index]
            writer.writerow([actual_label, predicted_label])
