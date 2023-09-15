import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from ydata_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer, SimpleImputer
from ydata_profiling import ProfileReport
import seaborn as sns
import matplotlib.pyplot as plt

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

for df in df_list:
    # Process 'duration_ms'
    average_duration = df[df['duration_ms'] != -1]['duration_ms'].mean()
    genre_name = df['genre'].iloc[0]
    df['duration_ms'].replace(-1, average_duration, inplace=True)
    
    # Process 'tempo'
    df['tempo'] = pd.to_numeric(df['tempo'], errors='coerce')
    average_tempo = df['tempo'].mean()
    df['tempo'].fillna(average_tempo, inplace=True)
    
    # Save the modified DataFrame as a CSV file with genre-specific name
    filename = f"{genre_name}_mod.csv"
    #filename = f"test_mod.csv"
    df.to_csv(filename, index=False)

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
#df_list_imputed = [test_mod]




# uncomment if you want to create all data
df_csv_concat = pd.concat(df_list_imputed, ignore_index=True)
df_csv_concat.to_csv('training-data/alldata.csv', index=False)
all_data = pd.read_csv('training-data/alldata.csv')
# profile = ProfileReport(all_data, title="Profiling Report")
# profile.to_file("report.html")



#------------REMOVE OUTLIERS-----------------
columns_to_check = ['popularity', 'acousticness', 'danceability', 'duration_ms',
                    'energy', 'instrumentalness', 'liveness','loudness', 
                    'speechiness','valence' ] 
def detect_outliers_zscore(data, columns, threshold=2):
    z_scores = np.abs((data[columns] - data[columns].mean()) / data[columns].std())
    return z_scores > threshold
outliers = detect_outliers_zscore(all_data, columns_to_check)
print(all_data[outliers.any(axis=1)])
# plt.figure(figsize=(12, 6))
# plt.boxplot(all_data[columns_to_check], vert=False)
# plt.xticks(range(1, len(columns_to_check) + 1), columns_to_check, rotation=45)
# plt.xlabel('Columns')
# plt.ylabel('Values')
# plt.title('Boxplots of Columns with Outliers')
# plt.show()
#------------DROP IRRELEVANT-----------------
all_data = all_data.drop(columns=['track_id', 'track_name', 'artist_name', 'instance_id'])

#------------IMPUTE THE DATA-----------------
# imputer = SimpleImputer(strategy="mean")
# all_data['tempo'] = all_data['tempo'].replace('?', np.nan)
# all_data['tempo'] = pd.to_numeric(all_data['tempo'], errors='coerce')
# all_data['duration_ms'] = all_data['duration_ms'].replace(-1, np.nan)
# all_data['popularity'] = all_data['popularity'].replace(0, np.nan)
# all_data['tempo'] = imputer.fit_transform(all_data['tempo'].values.reshape(-1, 1))
# all_data['duration_ms'] = imputer.fit_transform(all_data['duration_ms'].values.reshape(-1, 1))
# all_data['popularity'] = imputer.fit_transform(all_data['popularity'].values.reshape(-1, 1))
# all_data['instrumentalness'] = all_data['instrumentalness'].replace(0, np.nan)
# all_data['instrumentalness'] = imputer.fit_transform(all_data['instrumentalness'].values.reshape(-1, 1))

# Define a function for KNN imputation
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
all_data = knn_impute_columns(all_data, columns_to_impute, k_neighbors)

#------------SCALE THE DATA-----------------
scaler = MinMaxScaler()
numerical_columns = ['popularity', 'acousticness', 'danceability',
       'duration_ms', 'energy', 'instrumentalness', 'liveness',
       'loudness', 'speechiness', 'tempo', 'valence']
all_data[numerical_columns] = scaler.fit_transform(all_data[numerical_columns])

#------------ENCODE THE DATA-----------------
label_encoder_mode = LabelEncoder()
label_encoder_time_signature = LabelEncoder()
label_encoder_key = LabelEncoder()
all_data['mode'] = label_encoder_mode.fit_transform(all_data['mode'])
all_data['time_signature'] = label_encoder_time_signature.fit_transform(all_data['time_signature'])
all_data['key'] = label_encoder_key.fit_transform(all_data['key'])


# all_data = pd.get_dummies(all_data, columns=['genre'], prefix='genre')
# correlations = all_data.corr()
# plt.figure(figsize=(10, 6))
# sns.heatmap(correlations, cmap='coolwarm', annot=True, linewidths=.5)
# plt.title('Correlation Heatmap: Features vs. Encoded Genres')
# plt.show()

all_data.to_csv('training-data/alldata_notempo.csv', index=False)

