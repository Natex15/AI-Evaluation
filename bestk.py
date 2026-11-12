import matplotlib.pyplot as plt
import preprocess
import split
import macrof1
from knn import KNN

def plot_elbow_k(max_k=20):
    processed_data = preprocess.preprocess('conflict_data_phl.csv')
    train, test = split.split_dataset(processed_data)

    train = preprocess.classify_groups(train)
    test = preprocess.classify_groups(test)
    test_actual = preprocess.get_target(test)

    k_values = []
    f1_scores = []

    for k in range(1, max_k + 1):
        knn_model = KNN(k=k, p=2)
        knn_model.fit(train)
        predictions = knn_model.predict(test)

        score = macrof1.macrof1(test_actual, predictions)
        k_values.append(k)
        f1_scores.append(score)
        print(f"k = {k}: Macro F1 = {score:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, f1_scores, marker='o', linestyle='-', color='b')
    plt.title('KNN Elbow Method using Macro F1 Score')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Macro F1 Score')
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    plot_elbow_k()

# Based on this code the best k is 3