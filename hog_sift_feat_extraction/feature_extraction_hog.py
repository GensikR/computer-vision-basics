import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


def extract_hog_features():
    # Load the CIFAR-10 dataset
    cifar = fetch_openml('CIFAR_10_small', version=1, parser='auto')

    # Convert the input data to a numpy array and reshape
    cifar_rgb = np.array(cifar.data, dtype='uint8')
    cifar_rgb = cifar_rgb.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    # Convert images to grayscale
    cifar_gray = rgb2gray(cifar_rgb)

    # Visualize the first 10 images
    fig, axes = plt.subplots(1, 10, figsize=(10, 1))
    for i, ax in enumerate(axes):
        ax.imshow(cifar_gray[i], cmap='gray')
        ax.axis('off')
    plt.show()

    # Initialize variables for HOG features and labels
    hog_features = []
    y_features = []

    # Track variables for summary
    total_features_extracted = 0

    # Process each image to extract HOG features
    for idx in tqdm(range(cifar_gray.shape[0]), desc="Processing images"):
        try:
            features = hog(cifar_gray[idx], pixels_per_cell=(8, 8), cells_per_block=(2, 2))
            hog_features.append(features)
            y_features.append(cifar.target[idx])  # Store label
            
            # Update total features extracted
            total_features_extracted += features.shape[0]
        except Exception as e:
            pass

    # Convert lists to numpy arrays
    hog_features_np = np.array(hog_features)
    y_features_np = np.array(y_features)

    return hog_features_np, y_features_np, total_features_extracted

    