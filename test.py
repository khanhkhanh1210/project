from ultralytics import YOLO
import cv2
import torch
import easyocr
import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure the GPU is available and set it as the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the trained YOLO model and move it to the GPU
model = YOLO('license_plate_v8/train/weights/best.pt')  # Load the trained YOLO model
model.to(device)  # Send YOLO model to GPU

# Initialize EasyOCR reader with GPU enabled
reader = easyocr.Reader(['en'], gpu=(device.type == 'cuda'))  # Use GPU for EasyOCR

# Path to the input image
image_path = 'plate_1.jpg'

# Perform object detection using YOLO
results = model.predict(image_path, save=True, device=device)  # Ensure inference runs on GPU

# Load the original image
image = cv2.imread(image_path)

# Directory to save cropped images and text results
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Extract bounding boxes and crop detected plates
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates (x1, y1, x2, y2)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
        cropped_image = image[y1:y2, x1:x2]  # Crop the detected plate region

# Save and print the cropped image
cropped_image_path = os.path.join(output_dir, f'cropped_plate_{i}.jpg')
cv2.imwrite(cropped_image_path, cropped_image)
print(f'Cropped image saved to {cropped_image_path}')
plt.figure()
plt.title(f'Cropped Image {i}')
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# **Apply Preprocessing**
# Resize the cropped image
resized_image = cv2.resize(cropped_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
# Convert to grayscale
grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# **Apply Highboost Filtering**
# Gaussian blur
blurred = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
# Compute mask
mask = grayscale_image - blurred
# Apply highboost filtering
A = 1  # Boost factor
highboost_image = grayscale_image + A * mask
# Normalize to uint8 for saving
highboost_image_uint8 = cv2.normalize(highboost_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Save and print the highboost processed image
highboost_image_path = os.path.join(output_dir, f'highboost_plate_{i}.jpg')
cv2.imwrite(highboost_image_path, highboost_image_uint8)
print(f'Highboost processed image saved to {highboost_image_path}')
plt.figure()
plt.title(f'Highboost Processed Image {i}')

plt.imshow(highboost_image_uint8, cmap='gray')
plt.axis('off')
plt.show()

# Perform OCR on the highboost processed image
text_results = reader.readtext(highboost_image_uint8, detail=0)  # Extract text only
print(f'Text extracted from plate {i}: {text_results}')

# Save the extracted text to individual text files
text_file_path = os.path.join(output_dir, f'cropped_plate_{i}.txt')
with open(text_file_path, 'w') as text_file:
    text_file.write('\n'.join(text_results))
    print(f'Text saved to {text_file_path}')