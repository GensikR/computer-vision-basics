import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


def evaluate_hog(hog_features_np, y_features_np, total_features_extracted):
# Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(hog_features_np, y_features_np, test_size=0.2, random_state=42)

    # Train an SVM classifier
    svm = LinearSVC(random_state=42)
    svm.fit(X_train, y_train)

    # Evaluate the model
    accuracy = svm.score(X_test, y_test)

    # Assuming correct matches are those that the SVM predicts correctly
    y_pred = svm.predict(X_test)
    total_correct_matches = np.sum(y_pred == y_test)

    # Print summary
    print(f'Total features extracted: {total_features_extracted}')
    print(f'Total correct matches: {total_correct_matches}')
    print(f'SVM accuracy with HOG features: {accuracy:.2f}')