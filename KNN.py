import pandas as pd
import numpy as np


def dataLoadingAndNormalization():
    data = pd.read_csv("data.csv")
    pd.set_option("display.max_columns", None)

    important_columns = [
        "track_popularity",
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms",
    ]

    refinedData = data[important_columns]

    columns_to_exclude = ["track_id", "track_popularity"]

    # Create a copy of the DataFrame to avoid modifying the original
    normalized_data = refinedData.copy()

    # Iterate through all columns in the DataFrame
    for column in refinedData.columns:
        if column not in columns_to_exclude:
            # Find the range (max - min) for the column
            column_min = refinedData[column].min()
            column_max = refinedData[column].max()
            column_range = column_max - column_min

            # Normalize the column: (value - min) / range
            if column_range != 0:  # Avoid division by zero
                normalized_data[column] = (
                    refinedData[column] - column_min
                ) / column_range
            else:
                normalized_data[column] = 0

    normalized_data["track_popularity"] = (
        refinedData["track_popularity"] >= 50
    ).astype(int)

    np.random.seed(60)  # so results can be replicated
    normalized_data["randomNum"] = np.random.randint(
        1, 10001, size=len(normalized_data)
    )
    normalized_data["category"] = pd.qcut(
        normalized_data["randomNum"], q=5, labels=[0, 1, 2, 3, 4]
    )

    return normalized_data


def calculateEuclideanDistance(row1, row2):
    row1 = row1.drop("track_popularity")
    row2 = row2.drop("track_popularity")
    row1 = row1.drop("category")
    row2 = row2.drop("category")
    row1 = row1.drop("randomNum")
    row2 = row2.drop("randomNum")
    row1 = row1.drop("original_index")
    row2 = row2.drop("original_index")
    return np.sqrt(np.sum((row1 - row2) ** 2))


def calculate_accuracy(predicted, actual):
    correct_predictions = sum(predicted == actual)  # Count the correct predictions
    total_predictions = len(actual)  # Total number of predictions
    accuracy = correct_predictions / total_predictions  # Accuracy formula
    return accuracy


def create_distance_matrix(normalized_data):
    num_rows = len(normalized_data)

    distance_matrix = np.zeros((num_rows, num_rows))

    for i in range(num_rows):
        for j in range(i, num_rows):

            row_i = normalized_data.iloc[i]
            row_j = normalized_data.iloc[j]

            dist = calculateEuclideanDistance(row_i, row_j)

            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return distance_matrix


def kNN(normalized_data):
    normalized_data["original_index"] = [i for i in range(len(normalized_data))]
    distanceMatrix = create_distance_matrix(normalized_data)
    print("distance matrix has been computed")

    kNNAccuracyStats = np.zeros((3, 5))

    for fold in range(5):
        train_data = (normalized_data[normalized_data["category"] != fold]).copy()
        test_data = (normalized_data[normalized_data["category"] == fold]).copy()

        for testRowIndex, testRow in test_data.iterrows():
            tmpTrainData = train_data.copy()
            tmpTrainData["EucliDistanceToCurrTestDatum"] = tmpTrainData[
                "original_index"
            ].map(lambda idx: distanceMatrix[int(testRow["original_index"]), idx])
            tmpTrainData = tmpTrainData.sort_values(
                by="EucliDistanceToCurrTestDatum", ascending=True
            )

            for k in [1, 3, 5]:
                count0 = 0
                count1 = 0
                for i in range(k):
                    if tmpTrainData.iloc[i]["track_popularity"] == 0:
                        count0 += 1
                    else:
                        count1 += 1
                if count0 > count1:
                    test_data.loc[testRowIndex, f"{k}NNPrediction"] = 0
                else:
                    test_data.loc[testRowIndex, f"{k}NNPrediction"] = 1

        for i, k in enumerate([1, 3, 5]):
            predicted_values = test_data[str(k) + "NNPrediction"]
            actual_values = test_data["track_popularity"]
            tp = sum(1 if (a == 1) and (b == 1) else 0 for a, b in zip(actual_values, predicted_values))
            fp = sum(1 if (a == 0) and (b == 1) else 0 for a, b in zip(actual_values, predicted_values))
            tn = sum(1 if (a == 0) and (b == 0) else 0 for a, b in zip(actual_values, predicted_values))
            fn = sum(1 if (a == 1) and (b == 0) else 0 for a, b in zip(actual_values, predicted_values))


            print(f"True Positives (TP): {tp}")
            print(f"False Positives (FP): {fp}")
            print(f"True Negatives (TN): {tn}")
            print(f"False Negatives (FN): {fn}")

            accuracy = calculate_accuracy(predicted_values, actual_values)
            print(f"Accuracy for fold = {fold}, k={k}: {accuracy:.4f}")
            kNNAccuracyStats[i][fold] = accuracy

    for i, k in enumerate([1, 3, 5]):
        mean_acc = np.mean(kNNAccuracyStats[i])
        std_acc = np.std(kNNAccuracyStats[i])

        print(
            f"k={k}: Mean Accuracy = {mean_acc:.4f}, Standard Deviation = {std_acc:.4f}"
        )


def main():
    normalizedData = dataLoadingAndNormalization()
    #normalizedData = normalizedData.sample(n=100, random_state=42)
    #print(normalizedData.index)
    normalizedData = normalizedData.head(100)
    print(sum(normalizedData["track_popularity"]))
    kNN(normalizedData)


# This ensures the main() function is called when the script is executed
if __name__ == "__main__":
    main()
