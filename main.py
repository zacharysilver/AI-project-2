import pandas as pd
import numpy as np

data = pd.read_csv('data.csv')
pd.set_option('display.max_columns', None)

important_columns = ['track_popularity', 'danceability', 
                     'energy', 'key', 'loudness', 'mode', 
                     'speechiness', 'acousticness', 'instrumentalness', 
                     'liveness', 'valence', 'tempo', 'duration_ms']

refinedData = data[important_columns]

columns_to_exclude = ['track_id', 'track_popularity']

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
            normalized_data[column] = (refinedData[column] - column_min) / column_range
        else:
            normalized_data[column] = 0

normalized_data['track_popularity'] = (refinedData['track_popularity'] >= 50).astype(int)

np.random.seed(90) 
normalized_data['randomNum'] = np.random.randint(1, 10001, size=len(normalized_data))
normalized_data['category'] = pd.qcut(normalized_data['randomNum'], q=5, labels=[0, 1, 2, 3, 4])

train_data = normalized_data[normalized_data['category'] < 4]  
test_data = normalized_data[normalized_data['category'] == 4]  

'''
train_data = train_data.drop(columns=['randomNum', 'category'])
test_data = test_data.drop(columns=['randomNum', 'category'])

print("Train Data Size:", train_data.shape)
print("Test Data Size:", test_data.shape)
print("normalized data size", normalized_data.shape)

print("Train Data:\n", train_data.head())
print("\nTest Data:\n", test_data.head())
'''

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

            #Compute summary statistics for each k value
            


                     

