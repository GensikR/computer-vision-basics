import numpy as np
from PIL import Image

def create_image_pyramid(image, pyramid_height):
    # Load the image as a PIL Image object
    pil_image = Image.fromarray(image)

    # Get the base filename without extension
    filename = "img"

    # Loop through the pyramid levels
    for i in range(1, pyramid_height + 1):
        # Calculate the scale factor
        scale_factor = 2 ** i

        # Resize the image using PIL
        resized_image = pil_image.resize((pil_image.width // scale_factor, pil_image.height // scale_factor))

        # Save the resized image with the scale factor appended to the filename
        resized_image.save(f"{filename}_{scale_factor}x.png")

