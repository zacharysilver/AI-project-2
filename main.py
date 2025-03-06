import pandas as pd

data = pd.read_csv('data.csv')
pd.set_option('display.max_columns', None)

important_columns = ['track_id', 'track_popularity', 'danceability', 
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
            # If the column has the same value for all rows, keep it as 0 (or 1 if you prefer)
            normalized_data[column] = 0

normalized_data['track_popularity'] = (refinedData['track_popularity'] >= 50).astype(int)

# Print the first 5 rows of the normalized data
print(normalized_data.head())


