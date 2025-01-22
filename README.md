# Movie Recommendation System

A Python-based movie recommendation system that suggests similar movies based on content features including keywords, cast, director, and genres. This project implements both keywords-based and description-based recommendation algorithms using natural language processing techniques.

## Features

- Content-based movie recommendation system
- Utilizes movie metadata including cast, director, genres, and keywords
- Implements text processing and vectorization using scikit-learn
- Calculates movie similarity using cosine similarity
- Handles missing data and provides graceful error handling

## Technologies Used

- Python 3.x
- pandas
- numpy
- scikit-learn
- Natural Language Processing (NLP)

## Dataset

This project uses "The Movies Dataset" which contains metadata for thousands of movies. The dataset includes information such as:
- Movie titles
- Cast and crew
- Plot keywords
- Genres
- Overview descriptions

## Installation

1. Clone the repository
    ```
    git clone https://github.com/ngyathin16/movie-recommendation-system.git
    ```

2. Install required packages
    ```
    pip install pandas numpy scikit-learn
    ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset source: [The Movies Dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset)
- Inspired by DataCamp's tutorial on building recommendation systems
