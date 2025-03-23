import pandas as pd
import numpy as np

data = pd.read_csv('data.csv')

data['fold'] = np.random.randint(0, 5, size=len(data))

train_data = data[data['fold'] < 4]  
test_data = data[data['fold'] == 4]  


train_data = train_data.drop(columns=['fold'])
test_data = test_data.drop(columns=['fold'])

print("Train Data Size:", train_data.shape)
print("Test Data Size:", test_data.shape)

