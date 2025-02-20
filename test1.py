import numpy as np
from PIL import Image

# Simulate the image acquisition process, generating an array of the same size as in your original image (RGBA)
# Assuming the original data size you have is 230400, with width=320 and height=180, in RGBA (4 channels)

def simulate_get_monocular_image():
    # Create an array that simulates the original image data coming from AirSim
    width = 320
    height = 180
    channels = 4  # RGBA

    # Create an RGBA image with random values (as a simulation)
    img_data = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)

    # Convert RGBA to RGB using PIL
    img = Image.fromarray(img_data, 'RGBA')
    img_rgb = img.convert('RGB')  # Now we have an image with shape (height, width, 3)
    print("Converted RGBA to RGB")

    # Convert the image back to a numpy array for resizing
    img_array_rgb = np.asarray(img_rgb)
    print(f"After conversion to RGB: {img_array_rgb.shape}")

    return img_array_rgb

def simulate_get_state():
    # Step 1: Get the RGB image from AirSim (simulated)
    camera_image = simulate_get_monocular_image()

    # Step 2: Resize the image to match the expected model input size (let's test resizing to both (180, 320) and (103, 103))
    try:
        # Resizing to (180, 320) as an example (as expected by some parts of the code)
        target_width = 320
        target_height = 180
        img_pil = Image.fromarray(camera_image)
        img_resized = img_pil.resize((target_width, target_height))

        # Convert the resized image back to a numpy array
        camera_image_resized = np.asarray(img_resized)
        print(f"Image resized to expected size: {camera_image_resized.shape}")

        # Normalizing the image to range [0, 1]
        camera_image_normalized = camera_image_resized.astype(np.float32) / 255.0
        print(f"Image after normalization: {camera_image_normalized.shape}, dtype: {camera_image_normalized.dtype}")

    except Exception as e:
        print("Error during resizing or normalization:", str(e))

# Run the test function
simulate_get_state()
