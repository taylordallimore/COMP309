import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from ydata_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

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

testing = pd.read_csv('test-data/testing.csv')

#df_list = [testing]
df_list = [alternative, blues, childrens, comedy, electronic, folk, hiphop, movie, ska, soul]

# if colum contains -1, drop the instance
for df in df_list:
    df.drop(df[df['duration_ms'] == -1].index, inplace=True)
    df.drop(df[df['tempo'] == "?"].index, inplace=True)
    df.drop(df[df['popularity'] == 0].index, inplace=True)
    
#df.to_csv('training-data/filtered_data.csv', index=False)
    
alternative_mod = pd.read_csv('alternative_mod.csv')
blues_mod = pd.read_csv('blues_mod.csv')
childrens_mod = pd.read_csv("Children's Music_mod.csv")
comedy_mod = pd.read_csv('comedy_mod.csv')
electronic_mod = pd.read_csv('electronic_mod.csv')
folk_mod = pd.read_csv('folk_mod.csv')
hiphop_mod = pd.read_csv('hip-hop_mod.csv')
movie_mod = pd.read_csv('movie_mod.csv')
ska_mod = pd.read_csv('ska_mod.csv')
soul_mod = pd.read_csv('soul_mod.csv')
test_mod = pd.read_csv('test_mod.csv')

df_list_imputed = [alternative_mod, blues_mod, childrens_mod, comedy_mod, electronic_mod, folk_mod, hiphop_mod, movie_mod, ska_mod, soul_mod]
df_csv_concat = pd.concat(df_list, ignore_index=True)


df_csv_concat.to_csv('training-data/alldata.csv', index=False)
alldata = pd.read_csv('training-data/alldata.csv')
good_data = alldata.drop(['track_id', 'track_name', 'artist_name', 'instance_id'], axis=1)


time_signature_encoder = LabelEncoder()
good_data['time_signature_encoded'] = time_signature_encoder.fit_transform(good_data['time_signature'])


# Create a LabelEncoder for the "key" column
key_encoder = LabelEncoder()
good_data['key_encoded'] = key_encoder.fit_transform(good_data['key'])

# Create a LabelEncoder for the "mode" column
mode_encoder = LabelEncoder()
good_data['mode_encoded'] = mode_encoder.fit_transform(good_data['mode'])

# Drop the original "key" and "mode" columns if needed
data = good_data.drop(columns=['key', 'mode', 'time_signature'])


# # data.replace('?', np.nan, inplace=True)
data.to_csv('training-data/encoded_data.csv', index=False)



# #Select only the numeric columns for scaling (exclude 'genre')
# numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
# print(numeric_columns)
# # Create a MinMaxScaler instance
# scaler = MinMaxScaler()
# # Fit the MinMaxScaler on the numeric columns in X
# scaler.fit(data[numeric_columns])
# # Transform the numeric columns in X using the fitted scaler
# data[numeric_columns] = scaler.transform(data[numeric_columns])
# # Display the updated DataFrame
# print(data.head())








# filtered_data = data[data['instrumentalness'] == 0]
# filtered_data.to_csv('training-data/filtered_data.csv', index=False)
# # Count the occurrences of each genre in the filtered DataFrame
# genre_counts = filtered_data['genre'].value_counts()

# # Print the genre counts
# print("Genre Counts for Rows with instrumentalness = 0:")
# print(genre_counts)

# ------- CHECKING INSTRUMENTALNESS ------------------------------

# filtered_data = alldata[alldata['instrumentalness'] == 0]
# filtered_data.to_csv('training-data/filtered_data.csv', index=False)
# # Count the occurrences of each genre in the filtered DataFrame
# genre_counts = filtered_data['genre'].value_counts()

# # Print the genre counts
# print("Genre Counts for Rows with instrumentalness = 0:")
# print(genre_counts)

# good_data.to_csv('training-data/good_data.csv', index=False)
# print(good_data.head())














