import pandas as pd
import numpy as np
data = pd.read_csv('data.csv')
pd.set_option('display.max_columns', None)

important_columns = ['track_popularity', 'danceability', 
                     'energy', 'key', 'loudness', 'mode', 
                     'speechiness', 'acousticness', 'instrumentalness', 
                     'liveness', 'valence', 'tempo', 'duration_ms']

data = data[important_columns]
data['track_popularity'] = (data['track_popularity'] >= 50).astype(int)
        


def bestEntropy(data):
    bestEntropy = 1
    bestColumn = ''
    bestValue = 0
    for column in data.columns:
        if column != 'track_popularity':
            sorted_data = data.sort_values(by=column)
            lt_true = 0
            lt_false = 0
            gt_true = sum(data['track_popularity'])
            gt_false = len(data) - gt_true
            for i in range(len(sorted_data)-1):
                if sorted_data.iloc[i]['track_popularity'] == 1:
                    gt_true -= 1
                    lt_true += 1
                else:
                    gt_false -= 1
                    lt_false += 1
                if i + 1 < len(sorted_data) and sorted_data.iloc[i][column] == sorted_data.iloc[i + 1][column]:
                    continue
                lt_entropy = 0
                gt_entropy = 0
                if lt_true != 0 and lt_false != 0:
                    lt_entropy = -((lt_true / (lt_true + lt_false)) * np.log2(lt_true / (lt_true + lt_false)) + (lt_false / (lt_true + lt_false)) * np.log2(lt_false / (lt_true + lt_false)))
                if gt_true != 0 and gt_false != 0:
                    gt_entropy = -((gt_true / (gt_true + gt_false)) * np.log2(gt_true / (gt_true + gt_false)) + (gt_false / (gt_true + gt_false)) * np.log2(gt_false / (gt_true + gt_false)))
                entropy = (lt_true + lt_false) / len(sorted_data) * lt_entropy + (gt_true + gt_false) / len(sorted_data) * gt_entropy
                if entropy < bestEntropy:
                    bestEntropy = entropy
                    bestColumn = column
                    bestValue = sorted_data.iloc[i][column]
    return bestColumn, bestValue, bestEntropy

def onelevel(data):
    column, val, entr = bestEntropy(data)

    print(column, val, entr)
    lt_true = data[data[column] < val]['track_popularity'].sum()
    lt_false = len(data[data[column] < val]) - lt_true
    gt_true = data[data[column] >= val]['track_popularity'].sum()
    gt_false = len(data[data[column] >= val]) - gt_true
    print(lt_true, lt_false, gt_true, gt_false)
    return column, val

def buildTree(data, depth):
    if depth ==0:
        return bool(np.array(data['track_popularity']).mean()>0.5)
    column, val = onelevel(data)
    left = data[data[column] < val]

    right = data[data[column] >= val]
    return (column, val, buildTree(left, depth -1), buildTree(right, depth -1))

def predict(tree, song):
    if type(tree) == bool:
        return tree
    column, val, left, right = tree
    if song[column] < val:
        return predict(left, song)
    else:
        return predict(right, song)

tree  = buildTree(data, 3)

true_positives, true_negatives, false_positives, false_negatives = 0, 0, 0, 0

for i in range(len(data)):
    song = data.iloc[i]
    prediction = predict(tree, song)
    if prediction and song['track_popularity']:
        true_positives += 1
    elif prediction and not song['track_popularity']:
        false_positives += 1
    elif not prediction and song['track_popularity']:
        false_negatives += 1
    else:
        true_negatives += 1

print(f"True positives: {true_positives}")
print(f"True negatives: {true_negatives}")
print(f"False positives: {false_positives}")
print(f"False negatives: {false_negatives}")

