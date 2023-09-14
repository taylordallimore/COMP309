import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from ydata_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# load in data and create all data
alternative = pd.read_csv('training-data/alternative.csv')
blues = pd.read_csv('training-data/blues.csv')
childrens = pd.read_csv('training-data/children_music.csv')
comedy = pd.read_csv('training-data/comedy.csv')
electronic = pd.read_csv('training-data/electronic.csv')
folk = pd.read_csv('training-data/folk.csv')
hiphop = pd.read_csv('training-data/hip-hop.csv')
movie = pd.read_csv('training-data/movie.csv')
ska = pd.read_csv('training-data/ska.csv')
soul = pd.read_csv('training-data/soul.csv')

df_list = [alternative, blues, childrens, comedy, electronic, folk, hiphop, movie, ska, soul]
df_csv_concat = pd.concat(df_list, ignore_index=True)
df_csv_concat.to_csv('training-data/alldata.csv', index=False)
all_data = pd.read_csv('training-data/alldata.csv')

#------------DROP IRRELEVANT-----------------
all_data = all_data.drop(columns=['track_id', 'track_name', 'artist_name', 'instance_id'])

#------------IMPUTE THE DATA-----------------
imputer = SimpleImputer(strategy="mean")
all_data['tempo'] = all_data['tempo'].replace('?', np.nan)
all_data['tempo'] = pd.to_numeric(all_data['tempo'], errors='coerce')
all_data['duration_ms'] = all_data['duration_ms'].replace(-1, np.nan)
all_data['popularity'] = all_data['popularity'].replace(0, np.nan)
all_data['tempo'] = imputer.fit_transform(all_data['tempo'].values.reshape(-1, 1))
all_data['duration_ms'] = imputer.fit_transform(all_data['duration_ms'].values.reshape(-1, 1))
all_data['popularity'] = imputer.fit_transform(all_data['popularity'].values.reshape(-1, 1))
all_data['instrumentalness'] = all_data['instrumentalness'].replace(0, np.nan)
all_data['instrumentalness'] = imputer.fit_transform(all_data['instrumentalness'].values.reshape(-1, 1))

#------------SCALE THE DATA-----------------
scaler = MinMaxScaler()
numerical_columns = ['popularity', 'acousticness', 'danceability',
       'duration_ms', 'energy', 'instrumentalness', 'liveness',
       'loudness', 'speechiness', 'tempo',
       'valence']
all_data[numerical_columns] = scaler.fit_transform(all_data[numerical_columns])

#------------ENCODE THE DATA-----------------
label_encoder_mode = LabelEncoder()
label_encoder_time_signature = LabelEncoder()
label_encoder_key = LabelEncoder()
all_data['mode'] = label_encoder_mode.fit_transform(all_data['mode'])
all_data['time_signature'] = label_encoder_time_signature.fit_transform(all_data['time_signature'])
all_data['key'] = label_encoder_key.fit_transform(all_data['key'])

all_data.to_csv('training-data/alldata_notempo.csv', index=False)



# import pandas as pd
# import numpy as np
# from scipy import stats
# from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# from sklearn.impute import SimpleImputer

# # Load your data
# all_data = pd.read_csv('training-data/alldata.csv')

# # Define the numerical columns
# numerical_columns = ['popularity', 'acousticness', 'danceability',
#                      'duration_ms', 'energy', 'instrumentalness', 'key', 'liveness',
#                      'loudness', 'mode', 'speechiness', 'tempo', 'time_signature',
#                      'valence']

# # Replace non-numeric values (e.g., '?') with NaN
# all_data[numerical_columns] = all_data[numerical_columns].apply(pd.to_numeric, errors='coerce')

# # Identify and handle outliers using z-scores
# z_scores = np.abs(stats.zscore(all_data[numerical_columns]))
# outlier_threshold = 3  # Adjust this threshold as needed
# outlier_mask = (z_scores > outlier_threshold).any(axis=1)
# all_data_clean = all_data[~outlier_mask]

# # Impute missing values (you can adjust the imputation strategy as needed)
# imputer = SimpleImputer(strategy="mean")
# all_data_clean[numerical_columns] = imputer.fit_transform(all_data_clean[numerical_columns])

# # Encode categorical variables
# label_encoder_mode = LabelEncoder()
# label_encoder_time_signature = LabelEncoder()
# label_encoder_key = LabelEncoder()
# all_data_clean['mode'] = label_encoder_mode.fit_transform(all_data_clean['mode'])
# all_data_clean['time_signature'] = label_encoder_time_signature.fit_transform(all_data_clean['time_signature'])
# all_data_clean['key'] = label_encoder_key.fit_transform(all_data_clean['key'])

# # Scale the data
# scaler = MinMaxScaler()
# all_data_clean[numerical_columns] = scaler.fit_transform(all_data_clean[numerical_columns])

# # Save the cleaned data
# all_data_clean.to_csv('training-data/alldata_cleaned.csv', index=False)