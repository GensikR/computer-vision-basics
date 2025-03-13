import numpy as np
from skimage.feature import SIFT
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

def evaluate_sift(sift_features, sift_features_np, y_features, num_features):
    # Build a vocabulary using KMeans clustering
    vocab_size = 100
    kmeans = KMeans(n_clusters=vocab_size, random_state=42)

    # Fit the KMeans model to the SIFT features
    kmeans.fit(sift_features_np)

    # Build histograms of cluster centers for each image
    image_histograms = []

    for feature in tqdm(sift_features, desc="Building histograms"):
        # Predict the closest cluster for each feature
        clusters = kmeans.predict(feature.reshape(feature.shape[0], -1))  # Reshape feature for prediction
        # Build a histogram of the clusters
        histogram, _ = np.histogram(clusters, bins=vocab_size, range=(0, vocab_size))
        image_histograms.append(histogram)

    # Convert histograms to numpy array
    image_histograms_np = np.array(image_histograms)

    # Adjust frequency using TF-IDF
    tfidf = TfidfTransformer()

    # Fit the TfidfTransformer to the histogram data
    tfidf.fit(image_histograms_np)

    # Transform the histogram data using the trained TfidfTransformer
    image_histograms_tfidf = tfidf.transform(image_histograms_np)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(image_histograms_tfidf, np.array(y_features, dtype=int), test_size=0.2, random_state=42)

    # Train an SVM classifier
    svm = LinearSVC(random_state=42)
    svm.fit(X_train, y_train)

    # Evaluate the model
    accuracy = svm.score(X_test, y_test)

    # Assuming correct matches are those that the SVM predicts correctly
    y_pred = svm.predict(X_test)
    total_correct_matches = np.sum(y_pred == y_test)

    # Print summary
    print(f'Total features extracted: {num_features}')
    print(f'Total correct matches: {total_correct_matches}')
    print(f'SVM accuracy: {accuracy:.2f}')