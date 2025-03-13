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


def extract_sift_features():
    # Load the CIFAR-10 dataset
    cifar = fetch_openml('CIFAR_10_small', version=1, parser='auto')

    # Convert the input data to a numpy array
    cifar_rgb = np.array(cifar.data, dtype='uint8')

    # Reshape the data to (num_images, height, width, num_channels)
    cifar_rgb = cifar_rgb.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    # Convert the images to grayscale
    cifar_gray = rgb2gray(cifar_rgb)

    # Visualize the first 10 images
    fig, axes = plt.subplots(1, 10, figsize=(10, 1))
    for i, ax in enumerate(axes):
        ax.imshow(cifar_gray[i], cmap='gray')
        ax.axis('off')
    plt.show()

    # Extract SIFT features
    sift = SIFT()
    sift_features = []
    y_features = []

    # Track variables for summary
    total_features_extracted = 0

    for idx in tqdm(range(cifar_gray.shape[0]), desc="Processing images"):
        try:
            sift.detect_and_extract(cifar_gray[idx])
            sift_features.append(sift.descriptors)
            y_features.append(cifar.target[idx])  # Only stores the label if the SIFT features are successfully extracted
            
            # Update total features extracted
            total_features_extracted += sift.descriptors.shape[0]
        except Exception as e:
            pass

    # Convert the list of SIFT features to a numpy array
    sift_features_np = np.concatenate(sift_features)

    return sift_features, sift_features_np, y_features, total_features_extracted


