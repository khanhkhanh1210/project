from ultralytics import YOLO
import cv2
import pytesseract
import os

# Specify the full path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load the trained YOLO model
model = YOLO('v8_best.pt')

# Perform object detection on the image
image_path = 'plate_1.jpg'
results = model.predict(image_path, save=True)

# Load the original image
image = cv2.imread(image_path)

# Directory to save cropped images and text results
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Custom Tesseract configuration for better OCR
custom_config = r'--oem 3 --psm 6'

# Extract bounding boxes and crop
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates (x1, y1, x2, y2)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
        cropped_image = image[y1:y2, x1:x2]  # Crop the region of interest

        # Save and debug the cropped image
        cropped_image_path = os.path.join(output_dir, f'cropped_plate_{i}.jpg')
        cv2.imwrite(cropped_image_path, cropped_image)
        print(f'Cropped image saved to {cropped_image_path}')

        # Display the cropped image (optional for debugging)
        # cv2.imshow(f"Cropped Image {i}", cropped_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Preprocess the cropped image for better OCR
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Apply thresholding
        resized_image = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  # Resize to improve accuracy
        cv2.imwrite("imageafterchange.png",gray)


    
        
        
        # Perform OCR on the processed image
        text = pytesseract.image_to_string(resized_image, config=custom_config, lang='eng')  # Set language to 'eng'
        print(f'Text extracted from image {i}: {text}')
        result = pytesseract.image_to_data(resized_image, output_type=pytesseract.Output.DICT)
        print(result)

        # Save the extracted text to individual text files
        text_file_path = os.path.join(output_dir, f'cropped_plate_{i}.txt')
        with open(text_file_path, 'w') as text_file:
            text_file.write(text)
        print(f'Text saved to {text_file_path}')

        # Save all extracted text into one combined file
        # combined_text_file_path = '/Users/doand/OneDrive/Desktop/pythonproj/myenv/project-test/imagetotxt/txtafterconvert.txt'
        # with open(combined_text_file_path, 'a') as combined_file:
        #     combined_file.write(f"Text from image {i}:\n{text}\n")
        # print(f"Text from image {i} appended to {combined_text_file_path}")





# import matplotlib.pyplot as plt

# # Load the image
# img = cv2.imread('/Users/doand/OneDrive/Desktop/pythonproj/myenv/project-test/imagetotxt/images.png')  # Load as grayscale

# # Calculate the histogram
# #hist = cv2.calcHist([img], [0], None, [256], [0, 256])
# #show histogram
# #plt.plot(hist)
# #plt.show()





# import cv2
# import numpy as np

# #img = cv2.imread('/Users/doand/OneDrive/Desktop/pythonproj/myenv/project-test/imagetotxt/images.png', -1)

# rgb_planes = cv2.split(img)

# result_planes = []
# result_norm_planes = []
# for plane in rgb_planes:
#     dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
#     bg_img = cv2.medianBlur(dilated_img, 21)
#     diff_img = 255 - cv2.absdiff(plane, bg_img)
#     norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
#     result_planes.append(diff_img)
#     result_norm_planes.append(norm_img)
    
# result = cv2.merge(result_planes)
# result_norm = cv2.merge(result_norm_planes)

# cv2.imwrite('/Users/doand/OneDrive/Desktop/pythonproj/myenv/project-test/imagetotxt/shadowimage.png', result)
# cv2.imwrite('/Users/doand/OneDrive/Desktop/pythonproj/myenv/project-test/imagetotxt/shadowimage(1).png', result_norm)


# # Calculate the histogram
# hist1 = cv2.calcHist([img], [0], None, [256], [0, 256])

# #show histogram
# plt.plot(hist1)
# plt.show()