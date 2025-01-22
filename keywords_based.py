import kagglehub
import pandas as pd
import os
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download latest version
path = kagglehub.dataset_download("rounakbanik/the-movies-dataset")

# Load Movies MetaData
file_path = os.path.join(path, "movies_metadata.csv")
metadata = pd.read_csv(file_path, low_memory=False)

# Load credits
file_path = os.path.join(path, "credits.csv")
credits = pd.read_csv(file_path, low_memory=False)

# Load credits
file_path = os.path.join(path, "keywords.csv")
keywords = pd.read_csv(file_path, low_memory=False)

# Remove rows with bad IDs.
metadata = metadata.drop([19730, 29503, 35587])

# Convert IDs to int. Required for merging
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
metadata['id'] = metadata['id'].astype('int')

# Merge keywords and credits into your main metadata dataframe
metadata = metadata.merge(credits, on='id')
metadata = metadata.merge(keywords, on='id')

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(literal_eval)

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []

# Define new director, cast, genres and keywords features that are in a suitable form.
metadata['director'] = metadata['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(get_list)

# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

# Apply clean_data function to your features.
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    metadata[feature] = metadata[feature].apply(clean_data)

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

# Fill NaN values in director with empty string
metadata['director'] = metadata['director'].fillna('')

# Create the soup
metadata['soup'] = metadata.apply(create_soup, axis=1)

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(metadata['soup'])

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

# Reset index of your main DataFrame and construct reverse mapping as before
metadata = metadata.reset_index()
indices = pd.Series(metadata.index, index=metadata['title'])

def get_recommendations(title, cosine_sim):
    try:
        # Get the index of the movie
        idx = indices[title]
        
        # Get the pairwise similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Sort movies based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get scores of the 10 most similar movies (excluding itself)
        sim_scores = sim_scores[1:11]
        
        # Get movie indices
        movie_indices = [i[0] for i in sim_scores]
        
        # Ensure indices are within bounds of the metadata DataFrame
        valid_indices = [i for i in movie_indices if i < len(metadata)]
        
        if not valid_indices:
            print(f"No valid recommendations found for '{title}'")
            return pd.Series()
            
        return metadata['title'].iloc[valid_indices]
        
    except KeyError:
        print(f"Movie '{title}' not found. Similar titles:")
        similar_titles = indices.index[indices.index.str.contains(title, case=False, na=False)]
        if len(similar_titles) > 0:
            print("\n".join(similar_titles[:5]))  # Show up to 5 similar titles
        else:
            print("No similar titles found")
        return pd.Series()

# Test the recommendations with error handling
title = 'The Dark Knight Rises'
recommendations = get_recommendations(title, cosine_sim2)
if not recommendations.empty:
    print(f"\nRecommendations for '{title}':")
    print(recommendations)
