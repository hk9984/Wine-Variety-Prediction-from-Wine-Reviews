# Wine-Variety-Prediction-from-Wine-Reviews

The dataset used in the project is the dataset “winemag-data_first150k.csv”, containing records of wine reviews. The aim of the project is to read the text description of the review of the wine, and then predicting the variety of wine. The project is done in phases:

• Data Preprocessing: The data is cleaned to remove unnecessary noise and outliers, making it easier for the machine learning model to learn and give the most accurate predictions.

• Vectorization: Using TF-IDF vectorizer and Word2Vec embeddings, each text description is converted into vectors of length 5000 and 100 respectively, for the machine learning model to learn on.

• Feature Extraction: The vectors extracted after implementing both TF-IDF and Word2Vec embeddings are used as features to be fed into the machine learning classifier.

• Machine Learning: RandomForest, K-Nearest Neighbor and SVM classifiers have been used for the prediction and their performances have been evaluated.
