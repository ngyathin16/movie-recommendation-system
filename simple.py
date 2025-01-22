import kagglehub
import pandas as pd
import os

# Download latest version
path = kagglehub.dataset_download("rounakbanik/the-movies-dataset")

# Load Movies MetaData
file_path = os.path.join(path, "movies_metadata.csv")
metadata = pd.read_csv(file_path, low_memory=False)

# Print the first three rows
# print(metadata.head(3))

# Calculate mean of vote average column
C = metadata['vote_average'].mean()
# print(C)

# Calculate the minimum number of votes required to be in the chart, m
m = metadata['vote_count'].quantile(0.90)
# print(m)

# Filter out all qualified movies into a new DataFrame
q_movies = metadata.copy().loc[metadata['vote_count'] >= m]

# Function that computes the weighted rating of each movie
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

# Define a new feature 'score' and calculate its value with `weighted_rating()`
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

#Sort movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending=False)

#Print the top 15 movies
print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(20))
