import pandas as pd
from sklearn.ensemble import RandomForestClassifier
<<<<<<< Updated upstream
from sklearn.preprocessing import LabelEncoder
=======
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
>>>>>>> Stashed changes

# Load the test data
test_data = pd.read_csv('test-data/testing.csv')

<<<<<<< Updated upstream
# Preprocess 'duration_ms'
average_duration = test_data[test_data['duration_ms'] != -1]['duration_ms'].mean()
print("duration", average_duration)
test_data['duration_ms'].replace(-1, average_duration, inplace=True)


# Preprocess 'tempo'
test_data['tempo'] = pd.to_numeric(test_data['tempo'], errors='coerce')
average_tempo = test_data['tempo'].mean()
print("tempo", average_tempo)
test_data['tempo'].fillna(average_tempo, inplace=True)



# Create a LabelEncoder for the "key" column
key_encoder = LabelEncoder()
test_data['key_encoded'] = key_encoder.fit_transform(test_data['key'])

# Create a LabelEncoder for the "mode" column
mode_encoder = LabelEncoder()
test_data['mode_encoded'] = mode_encoder.fit_transform(test_data['mode'])
test_data = test_data.drop(columns=['key', 'mode', 'track_id', 'track_name', 'artist_name', 'time_signature', 'instrumentalness'])
=======
# def remove_outliers_iqr(data, columns, threshold=1.5):
#     filtered_data = data.copy()
#     for column in columns:
#         Q1 = data[column].quantile(0)
#         Q3 = data[column].quantile(0.95)
#         IQR = Q3 - Q1
#         lower_bound = Q1 - threshold * IQR
#         upper_bound = Q3 + threshold * IQR
#         filtered_data = filtered_data[(filtered_data[column] >= lower_bound) & (filtered_data[column] <= upper_bound)]
#     return filtered_data

columns_to_check = ['popularity', 'acousticness', 'danceability', 'duration_ms',
                    'energy', 'instrumentalness', 'liveness','loudness', 
                    'speechiness','valence' ] 
# test_data = remove_outliers_iqr(test_data, columns_to_check)

#------------DROP IRRELEVANT-----------------
test_data = test_data.drop(columns=['track_id', 'track_name', 'artist_name'])

#------------IMPUTE THE DATA-----------------
# imputer = SimpleImputer(strategy="mean")
# test_data['tempo'] = test_data['tempo'].replace('?', np.nan)
# test_data['tempo'] = pd.to_numeric(test_data['tempo'], errors='coerce')
# test_data['duration_ms'] = test_data['duration_ms'].replace(-1, np.nan)
# test_data['popularity'] = test_data['popularity'].replace(0, np.nan)
# test_data['tempo'] = imputer.fit_transform(test_data['tempo'].values.reshape(-1, 1))
# test_data['duration_ms'] = imputer.fit_transform(test_data['duration_ms'].values.reshape(-1, 1))
# test_data['popularity'] = imputer.fit_transform(test_data['popularity'].values.reshape(-1, 1))
# test_data['instrumentalness'] = test_data['instrumentalness'].replace(0, np.nan)
# test_data['instrumentalness'] = imputer.fit_transform(test_data['instrumentalness'].values.reshape(-1, 1))

def knn_impute_columns(data, columns_to_impute, k_neighbors=5):
    imputed_data = data.copy()
    imputed_data[columns_to_impute] = imputed_data[columns_to_impute].replace('?', np.nan)
    imputed_data[columns_to_impute] = imputed_data[columns_to_impute].replace(-1, np.nan)
    imputed_data[columns_to_impute] = imputed_data[columns_to_impute].replace(0, np.nan)
    knn_imputer = KNNImputer(n_neighbors=k_neighbors)
    imputed_data[columns_to_impute] = knn_imputer.fit_transform(imputed_data[columns_to_impute])
    return imputed_data

columns_to_impute = ['tempo', 'duration_ms', 'popularity', 'instrumentalness']
k_neighbors = 5  
test_data = knn_impute_columns(test_data, columns_to_impute, k_neighbors)

#------------SCALE THE DATA-----------------
scaler = MinMaxScaler()
numerical_columns = ['popularity', 'acousticness', 'danceability',
       'duration_ms', 'energy', 'instrumentalness', 'liveness',
       'loudness', 'speechiness', 'tempo', 'valence']
test_data[numerical_columns] = scaler.fit_transform(test_data[numerical_columns])

#------------ENCODE THE DATA-----------------
label_encoder_mode = LabelEncoder()
label_encoder_time_signature = LabelEncoder()
label_encoder_key = LabelEncoder()
test_data['mode'] = label_encoder_mode.fit_transform(test_data['mode'])
test_data['time_signature'] = label_encoder_time_signature.fit_transform(test_data['time_signature'])
test_data['key'] = label_encoder_key.fit_transform(test_data['key'])



test_data.to_csv('training-data/testing_ready.csv', index=False)
>>>>>>> Stashed changes

test_data.to_csv('test_mod.csv', index=False)

