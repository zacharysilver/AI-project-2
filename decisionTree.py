import pandas as pd
import numpy as np

data = pd.read_csv("data.csv")
pd.set_option("display.max_columns", None)


data = data[
    [
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
]

data["track_popularity"] = (data["track_popularity"] >= 50).astype(int)


def bestEntropy(data):
    bestEntropy = 1
    bestColumn = ""
    bestValue = 0
    for column in data.columns:
        if column != "track_popularity":
            sorted_data = data.sort_values(by=column)
            lt_true = 0
            lt_false = 0
            ge_true = sum((data["track_popularity"]).astype(int))
            ge_false = len(data) - ge_true
            for i in range(1, len(sorted_data)):
                if sorted_data.iloc[i - 1]["track_popularity"]:
                    ge_true -= 1
                    lt_true += 1
                else:
                    ge_false -= 1
                    lt_false += 1
                if sorted_data.iloc[i - 1][column] == sorted_data.iloc[i][column]:
                    continue
                lt_entropy = 0
                gt_entropy = 0
                if lt_true != 0 and lt_false != 0:
                    lt_entropy = -(
                        (lt_true / (lt_true + lt_false))
                        * np.log2(lt_true / (lt_true + lt_false))
                        + (lt_false / (lt_true + lt_false))
                        * np.log2(lt_false / (lt_true + lt_false))
                    )
                if ge_true != 0 and ge_false != 0:
                    gt_entropy = -(
                        (ge_true / (ge_true + ge_false))
                        * np.log2(ge_true / (ge_true + ge_false))
                        + (ge_false / (ge_true + ge_false))
                        * np.log2(ge_false / (ge_true + ge_false))
                    )
                entropy = (lt_true + lt_false) / len(sorted_data) * lt_entropy + (
                    ge_true + ge_false
                ) / len(sorted_data) * gt_entropy
                if entropy < bestEntropy:
                    bestEntropy = entropy
                    bestColumn = column
                    bestValue = sorted_data.iloc[i][column]
    return bestColumn, bestValue, bestEntropy


def onelevel(data):
    column, val, entr = bestEntropy(data)
    if column == "":
        return column, val
    print(column, val, entr)
    lt_true = data[data[column] < val]["track_popularity"].sum()
    lt_false = len(data[data[column] < val]) - lt_true
    ge_true = data[data[column] >= val]["track_popularity"].sum()
    ge_false = len(data[data[column] >= val]) - ge_true
    print(lt_true, lt_false, ge_true, ge_false)
    return column, val


def buildTree(data, depth):
    if len(data) == 0:
        return False
    if depth == 0:
        return bool(np.array(data["track_popularity"]).mean() >= 0.5)
    column, val, entr = bestEntropy(data)
    if column == "":
        return bool(np.array(data["track_popularity"]).mean() >= 0.5)
    left = data[data[column] < val]

    right = data[data[column] >= val]
    return (column, val, buildTree(left, depth - 1), buildTree(right, depth - 1))


def predict(tree, song):
    if type(tree) == bool:
        return tree
    column, val, left, right = tree
    if song[column] < val:
        return predict(left, song)
    else:
        return predict(right, song)


tree = buildTree(data, 3)
print(tree)

true_positives, true_negatives, false_positives, false_negatives = 0, 0, 0, 0

for i in range(len(data)):
    song = data.iloc[i]
    prediction = predict(tree, song)
    if prediction and song["track_popularity"] == 1:
        true_positives += 1
    elif prediction and not song["track_popularity"] == 1:
        false_positives += 1
    elif not prediction and song["track_popularity"] == 1:
        false_negatives += 1
    else:
        true_negatives += 1

print(f"True positives: {true_positives}")
print(f"True negatives: {true_negatives}")
print(f"False positives: {false_positives}")
print(f"False negatives: {false_negatives}")

accuracy = (true_positives + true_negatives) / (
    true_positives + true_negatives + false_positives + false_negatives
)
print(f"Accuracy: {accuracy}")
