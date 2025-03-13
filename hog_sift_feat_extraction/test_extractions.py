from feature_extraction_sift import extract_sift_features
from feature_extraction_hog import extract_hog_features
from evaluate_sift import evaluate_sift
from evaluate_hog import evaluate_hog

def test_hog():
    hog_features_np, y_features_np, total_features_extracted = extract_hog_features() 
    evaluate_hog(hog_features_np, y_features_np, total_features_extracted)

def test_sift():
    sift_features, sift_features_np, y_features, total_features_extracted = extract_sift_features()
    evaluate_sift(sift_features, sift_features_np, y_features, total_features_extracted)

def main():
    while True:
        user_input = input("Press 1 to test HOG feature extraction,\nPress 2 to test SIFT feature extraction\nAny other key to exit: ")
        if user_input == "1":
            test_hog()
        elif user_input == "2":
            test_sift()
        else:
            break

if __name__ == "__main__":
    main()
