import sys
import numpy as np
from PIL import Image

def main():
    if len(sys.argv) != 5:
        print("Usage: python program.py <filename> <hue_modification> <saturation_modification> <value_modification>")
        sys.exit(1)
    
    filename = sys.argv[1]
    hue_mod = float(sys.argv[2])
    sat_mod = float(sys.argv[3])
    val_mod = float(sys.argv[4])
    
    # Check if modifications are within range
    if not (0 <= hue_mod <= 1) or not (-1 <= sat_mod <= 1) or not (-1 <= val_mod <= 1):
        print("Error: Modifications out of range.")
        sys.exit(1)
    
    # Load the image
    img = np.array(Image.open(filename))
    
    # Convert RGB image to HSV
    hsv_img = rgb2hsv(img)
    
    # Modify the HSV image
    hsv_img[:, :, 0] = np.clip(hsv_img[:, :, 0] + hue_mod, 0, 1)
    hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1] + sat_mod, 0, 1)
    hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2] + val_mod, 0, 1)
    
    # Convert HSV image back to RGB
    modified_img = hsv2rgb(hsv_img)
    
    # Save the modified image
    Image.fromarray(modified_img.astype(np.uint8)).save("modified_" + filename)

def normalize_image(img):
    # Normalize the image if it is not already normalized
    if np.max(img) > 1:
        img = img / 255.0
    return img

def rgb2hsv(img):
    # Ensure the image is normalized
    img = normalize_image(img)
    
    # Get the R, G, and B values
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    # Get the maximum and minimum values
    max_val = np.max(img, axis=2)
    min_val = np.min(img, axis=2)
    delta = max_val - min_val

    # Calculate the value
    V = max_val

    # Calculate the saturation
    S = np.zeros_like(max_val)
    S[max_val != 0] = delta[max_val != 0] / max_val[max_val != 0]

    # Calculate the hue
    H = np.zeros_like(max_val)
    mask = (max_val == R) & (delta != 0)
    H[mask] = (G[mask] - B[mask]) / delta[mask] % 6
    mask = (max_val == G) & (delta != 0)
    H[mask] = (B[mask] - R[mask]) / delta[mask] + 2
    mask = (max_val == B) & (delta != 0)
    H[mask] = (R[mask] - G[mask]) / delta[mask] + 4
    H = (H / 6) % 1

    # Stack the channels
    hsv_img = np.stack((H, S, V), axis=2)
    return hsv_img

def hsv2rgb(hsv_img):
    # Get the hue, saturation, and value channels
    H = hsv_img[:, :, 0]
    S = hsv_img[:, :, 1]
    V = hsv_img[:, :, 2]

    # Calculate the chroma
    C = V * S

    # Calculate the intermediate values
    X = C * (1 - np.abs((H * 6) % 2 - 1))
    m = V - C

    # Initialize the RGB channels
    R = np.zeros_like(H)
    G = np.zeros_like(H)
    B = np.zeros_like(H)

    # Set the RGB values based on the hue
    mask = (0 <= H) & (H < 1/6)
    R[mask] = C[mask]
    G[mask] = X[mask]
    B[mask] = 0

    mask = (1/6 <= H) & (H < 2/6)
    R[mask] = X[mask]
    G[mask] = C[mask]
    B[mask] = 0

    mask = (2/6 <= H) & (H < 3/6)
    R[mask] = 0
    G[mask] = C[mask]
    B[mask] = X[mask]

    mask = (3/6 <= H) & (H < 4/6)
    R[mask] = 0
    G[mask] = X[mask]
    B[mask] = C[mask]

    mask = (4/6 <= H) & (H < 5/6)
    R[mask] = X[mask]
    G[mask] = 0
    B[mask] = C[mask]

    mask = (5/6 <= H) & (H <= 1)
    R[mask] = C[mask]
    G[mask] = 0
    B[mask] = X[mask]

    # Add the intermediate values to the RGB channels
    R = (R + m) * 255
    G = (G + m) * 255
    B = (B + m) * 255

    # Stack the channels
    rgb_img = np.stack((R, G, B), axis=2)
    return rgb_img



if __name__ == "__main__":
    main()
