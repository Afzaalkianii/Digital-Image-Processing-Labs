#IMPORTING LIBRARIES
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv.imread("sample-image.jpeg")

# Check if image exists
if image is None:
    raise FileNotFoundError("Image not found!")

# Print image information
print("Image Shape:", image.shape)
print("Image Data Type:", image.dtype)

#STEP 03
# Print height, width and channels
height, width, channels = image.shape
print("Image Height:", height)
print("Image Width:", width)
print("Number of Channels:", channels)
# Print top-left pixel
print("Top-Left Pixel:", image[0, 0])
# Print center pixel
center_pixel = image[height // 2, width // 2]
print("Center Pixel:", center_pixel)
# Print global minimum and maximum
print("Global Minimum:", np.min(image))
print("Global Maximum:", np.max(image))
# Print per-channel minimum and maximum
print("Blue Channel Min:", np.min(image[:, :, 0]))
print("Blue Channel Max:", np.max(image[:, :, 0]))
print("Green Channel Min:", np.min(image[:, :, 1]))
print("Green Channel Max:", np.max(image[:, :, 1]))
print("Red Channel Min:", np.min(image[:, :, 2]))
print("Red Channel Max:", np.max(image[:, :, 2]))

# STEP 04
# Convert BGR to RGB
rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
# Convert image to Grayscale
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# Separate Red, Green and Blue channels
red_channel = rgb_image[:, :, 0]
green_channel = rgb_image[:, :, 1]
blue_channel = rgb_image[:, :, 2]
# Create a 2 x 3 figure
plt.figure(figsize=(12, 8))
# Display RGB image
plt.subplot(2, 3, 1)
plt.imshow(rgb_image)
plt.title("RGB Image")
plt.axis("off")
# Display Red channel
plt.subplot(2, 3, 2)
plt.imshow(red_channel, cmap="gray")
plt.title("Red Channel")
plt.axis("off")
# Display Green channel
plt.subplot(2, 3, 3)
plt.imshow(green_channel, cmap="gray")
plt.title("Green Channel")
plt.axis("off")
# Display Blue channel
plt.subplot(2, 3, 4)
plt.imshow(blue_channel, cmap="gray")
plt.title("Blue Channel")
plt.axis("off")
# Display Grayscale image
plt.subplot(2, 3, 5)
plt.imshow(gray_image, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
# Display original BGR image without conversion
plt.subplot(2, 3, 6)
plt.imshow(image)
plt.title("BGR Image (Unconverted)")
plt.axis("off")
# Adjust layout and show figure
plt.tight_layout()
plt.show()
# Get center pixel RGB value
center_rgb = rgb_image[height // 2, width // 2]
# Convert RGB to HEX
hex_color = "#{:02X}{:02X}{:02X}".format(
    int(center_rgb[0]),
    int(center_rgb[1]),
    int(center_rgb[2])
)
# Print RGB and HEX values
print("Center Pixel RGB:", center_rgb)
print("Center Pixel HEX:", hex_color)

# STEP 05
# Crop an exact 100 x 100 region
cropped_image = image[100:200, 100:200]

# Print crop shape
print("Cropped Image Shape:", cropped_image.shape)

# Resize crop to 200 x 200 pixels
resized_image = cv.resize(cropped_image, (200, 200))

# Convert images from BGR to RGB for Matplotlib
original_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
cropped_rgb = cv.cvtColor(cropped_image, cv.COLOR_BGR2RGB)
resized_rgb = cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)

# Display original, cropped and resized images side by side
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(original_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(cropped_rgb)
plt.title("Cropped Image (100x100)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(resized_rgb)
plt.title("Resized Image (200x200)")
plt.axis("off")

plt.tight_layout()
plt.show()

# Convert resized RGB image back to BGR
resized_bgr = cv.cvtColor(resized_rgb, cv.COLOR_RGB2BGR)

# Save the resized crop
save_status = cv.imwrite("resized_crop.png", resized_bgr)

# Print whether the file was saved successfully
print("File Saved Successfully:", save_status)

#STEP 06
# Flatten the grayscale image
flattened_image = gray_image.flatten()

# Plot a 256-bin histogram (range 0–256)
plt.figure(figsize=(8, 5))
plt.hist(flattened_image, bins=256, range=(0, 256))

# Add title and axis labels
plt.title("Grayscale Image Histogram")
plt.xlabel("Pixel Intensity (0–255)")
plt.ylabel("Number of Pixels")

# Save the histogram plot
plt.savefig("grayscale_histogram.png")

# Display the histogram
plt.show()

print("Histogram saved successfully as grayscale_histogram.png")

#STEP 07
# Create an 8-bit horizontal grayscale gradient (0 to 255)
gradient = np.linspace(0, 255, 256, dtype=np.uint8)
gradient = np.tile(gradient, (50, 1))

# Quantize gradient to 1-bit
gradient_1bit = (gradient // 128) * 255

# Quantize gradient to 2-bit (4 intensity levels)
gradient_2bit = (gradient // 64) * 85

# Quantize gradient to 4-bit (16 intensity levels)
gradient_4bit = (gradient // 16) * 17

# Display original and quantized gradients
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.imshow(gradient, cmap="gray", vmin=0, vmax=255)
plt.title("Original 8-bit Gradient (256 Levels)")
plt.axis("off")

plt.subplot(4, 1, 2)
plt.imshow(gradient_1bit, cmap="gray", vmin=0, vmax=255)
plt.title("1-bit Gradient (2 Levels)")
plt.axis("off")

plt.subplot(4, 1, 3)
plt.imshow(gradient_2bit, cmap="gray", vmin=0, vmax=255)
plt.title("2-bit Gradient (4 Levels)")
plt.axis("off")

plt.subplot(4, 1, 4)
plt.imshow(gradient_4bit, cmap="gray", vmin=0, vmax=255)
plt.title("4-bit Gradient (16 Levels)")
plt.axis("off")

plt.tight_layout()
plt.show()

# Print unique-value counts
print("8-bit Unique Values:", len(np.unique(gradient)))
print("1-bit Unique Values:", len(np.unique(gradient_1bit)))
print("2-bit Unique Values:", len(np.unique(gradient_2bit)))
print("4-bit Unique Values:", len(np.unique(gradient_4bit)))