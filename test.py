import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the test data
test_data = pd.read_csv('test-data/testing.csv')

#------------DROP IRRELEVANT-----------------
test_data = test_data.drop(columns=['track_id', 'track_name', 'artist_name'])

#------------IMPUTE THE DATA-----------------
imputer = SimpleImputer(strategy="mean")
test_data['tempo'] = test_data['tempo'].replace('?', np.nan)
test_data['tempo'] = pd.to_numeric(test_data['tempo'], errors='coerce')
test_data['duration_ms'] = test_data['duration_ms'].replace(-1, np.nan)
test_data['popularity'] = test_data['popularity'].replace(0, np.nan)
test_data['tempo'] = imputer.fit_transform(test_data['tempo'].values.reshape(-1, 1))
test_data['duration_ms'] = imputer.fit_transform(test_data['duration_ms'].values.reshape(-1, 1))
test_data['popularity'] = imputer.fit_transform(test_data['popularity'].values.reshape(-1, 1))
test_data['instrumentalness'] = test_data['instrumentalness'].replace(0, np.nan)
test_data['instrumentalness'] = imputer.fit_transform(test_data['instrumentalness'].values.reshape(-1, 1))

#------------SCALE THE DATA-----------------
scaler = MinMaxScaler()
numerical_columns = ['popularity', 'acousticness', 'danceability',
       'duration_ms', 'energy', 'instrumentalness', 'liveness',
       'loudness', 'speechiness', 'tempo',
       'valence']
test_data[numerical_columns] = scaler.fit_transform(test_data[numerical_columns])

#------------ENCODE THE DATA-----------------
label_encoder_mode = LabelEncoder()
label_encoder_time_signature = LabelEncoder()
label_encoder_key = LabelEncoder()
test_data['mode'] = label_encoder_mode.fit_transform(test_data['mode'])
test_data['time_signature'] = label_encoder_time_signature.fit_transform(test_data['time_signature'])
test_data['key'] = label_encoder_key.fit_transform(test_data['key'])



test_data.to_csv('training-data/testing_ready.csv', index=False)


