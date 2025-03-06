import pandas as pd
import numpy as np

def calclulateEuclideanDistance(row1, row2):
    row1 = row1.drop('track_popularity')
    row2 = row2.drop('track_popularity')
    return np.sqrt(np.sum((row1 - row2) ** 2))


def kNN():
        for fold in range(5):
            train_data = normalized_data[normalized_data['category'] != fold]  
            test_data = normalized_data[normalized_data['category'] == fold]  

            for index, testRow in test_data.iterrows():
                tmpTrainData = train_data.copy()
                tmpTrainData['EucliDistanceToCurrTestDatum'] = calclulateEuclideanDistance(testRow, tmpTrainData)
                tmpTrainData = tmpTrainData.sort_values(by='EucliDistanceToCurrTestDatum', ascending=True)

                for k in [1,3,5]:
                    count0 = 0
                    count1 = 0
                    for i in range(k+1):
                        if tmpTrainData["track_popularity"] == 0:
                            count0 += 1
                        else:
                            count1 += 1
                    if count0 > count1:
                        train_data[str(k) + "NNPrediction"] = 0
                    else:
                        train_data[str(k) + "NNPrediction"] = 1