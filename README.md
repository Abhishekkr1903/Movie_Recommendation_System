** Movie Recommendation System**

This project is a content-based movie recommendation system that suggests movies similar to a user’s input using cosine similarity on TF-IDF feature vectors.

** Project Overview**

The recommendation is based on:

User’s previous watch patterns (similar genre, cast, director, etc.)

Popular movies (top trends on platforms like Netflix)

Group-based recommendations (people with similar watch habits)

** How It Works**

Data Collection → Load movie dataset with details like genre, cast, director, keywords, tagline, etc.

Data Preprocessing → Clean and prepare data for analysis.

Feature Extraction → Convert textual information into numerical feature vectors using TF-IDF Vectorizer.

Similarity Calculation → Use Cosine Similarity to find movies most similar to the user’s choice.

Recommendation → Return a list of top similar movies.

** Technologies Used**

Python

Pandas, NumPy

Scikit-learn (TfidfVectorizer, cosine_similarity)

Difflib (for handling spelling variations in user input)
