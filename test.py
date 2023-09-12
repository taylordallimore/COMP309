import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the test data
test_data = pd.read_csv('test-data/testing.csv')

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

test_data.to_csv('test_mod.csv', index=False)

