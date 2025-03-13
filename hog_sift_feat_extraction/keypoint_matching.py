import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.feature import SIFT, match_descriptors
from matplotlib.patches import ConnectionPatch

# Function to load an image and convert it to grayscale
def load_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return image

# Function to detect SIFT keypoints and extract descriptors
def sift_features(dst_img, src_img):
    detector1 = SIFT()
    detector2 = SIFT()
    detector1.detect_and_extract(dst_img)
    detector2.detect_and_extract(src_img)
    keypoints1 = detector1.keypoints
    descriptors1 = detector1.descriptors
    keypoints2 = detector2.keypoints
    descriptors2 = detector2.descriptors  # Extract keypoint coordinates
    return keypoints1, descriptors1, keypoints2, descriptors2

# Custom descriptor matching function with Lowe's ratio test
def match_keypoints(descriptors1, descriptors2, ratio=0.7):
    matches = []

    # Brute-force matching
    bf = cv2.BFMatcher(cv2.NORM_L2)
    all_matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply Lowe's ratio test
    for m, n in all_matches:
        if m.distance < ratio * n.distance:
            matches.append([m.queryIdx, m.trainIdx])

    # Convert to numpy array for compatibility with further processing
    matches = np.array(matches)

    return matches

def plot_matches(dst_img, src_img, keypoints1, keypoints2, matches):
    # Select the points in img1 that match with img2 and vice versa
    print(matches.shape)
    dst = keypoints1[matches[:, 0]]
    src = keypoints2[matches[:, 1]]

    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(dst_img, cmap='gray')
    ax2.imshow(src_img, cmap='gray')

    for i in range(src.shape[0]):
        coordB = [dst[i][1], dst[i][0]]  # Adjusted indexing for structured arrays or tuples
        coordA = [src[i][1], src[i][0]]  # Adjusted indexing for structured arrays or tuples
        con = ConnectionPatch(xyA=coordA, xyB=coordB, coordsA="data", coordsB="data",
                              axesA=ax2, axesB=ax1, color="red")
        ax2.add_artist(con)
        ax1.plot(dst[i][1], dst[i][0], 'ro')  # Adjusted indexing for structured arrays or tuples
        ax2.plot(src[i][1], src[i][0], 'ro')  # Adjusted indexing for structured arrays or tuples

    ax1.axis('off')
    ax2.axis('off')
    plt.show()

# Main function to run keypoint matching
def keypoint_matching(image1_path, image2_path, match_ratio=0.7):
    # Load images
    image1 = load_image(image1_path)
    image2 = load_image(image2_path)

    # Detect keypoints and extract descriptors
    keypoints1, descriptors1, keypoints2, descriptors2 = sift_features(image1, image2)
    print(match_descriptors(descriptors1, descriptors2).shape)
    # Match descriptors
    matches_custom = match_keypoints(descriptors1, descriptors2)

    # Visualize matches
    plot_matches(image1, image2, keypoints1, keypoints2, matches_custom)

# Example usage
if __name__ == '__main__':
    keypoint_matching('yosemite1.jpg', 'yosemite2.jpg')
