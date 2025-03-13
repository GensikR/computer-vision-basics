import img_transforms as transforms
import create_img_pyramid as pyramid
import numpy as np
from PIL import Image

# Define the load_image and save_image functions here

def main():
    # Ask the user for the path to the image
    image_path = input("Enter the path to the image: ")
    img = load_image(image_path)

    while True:
        # Display the menu
        print("\nChoose an option:")
        print("1. Test random crop")
        print("2. Test extract patches")
        print("3. Test resize image")
        print("4. Test color jitter")
        print("5. Create image pyramid")
        print("6. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            # Test random crop
            size = int(input("Enter the crop size: "))
            new_img = transforms.random_crop(img, size)
            save_image(new_img, "random_crop.jpg")
            print("Image saved as random_crop.jpg")

        elif choice == "2":
            # Test extract patches
            num_patches = int(input("Enter the number of patches: "))
            patches = transforms.extract_patches(img, num_patches)
            # Save each patch
            for i, patch in enumerate(patches):
                save_image(patch, f"patch_{i}.jpg")
            print("Patches extracted")

        elif choice == "3":
            # Test resize image
            factor = float(input("Enter the scale factor: "))
            new_img = transforms.resize_img(img, factor)
            Image.fromarray(new_img.astype(np.uint8)).save("resized_image.jpg")
            #save_image(new_img, "resized_image.jpg")
            print("Image resized and saved as resized_image.jpg")

        elif choice == "4":
            # Test color jitter
            hue = float(input("Enter the maximum hue modification: "))
            saturation = float(input("Enter the maximum saturation modification: "))
            value = float(input("Enter the maximum value modification: "))
            new_img = transforms.color_jitter(img, hue, saturation, value)
            save_image(new_img, "color_jitter.jpg")
            print("Color jitter applied and image saved as color_jitter.jpg")

        elif choice == "5":
            # Create image pyramid
            pyramid_height = int(input("Enter the pyramid height: "))
            pyramid.create_image_pyramid(img, pyramid_height)
            print("Image pyramid created")

        elif choice == "6":
            # Exit the program
            break

        else:
            print("Invalid choice. Please try again.")

def load_image(file_path):
    return np.array(Image.open(file_path))

def save_image(image, file_path):
    Image.fromarray(image).save(file_path)


if __name__ == "__main__":
    main()
