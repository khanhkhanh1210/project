import cv2
import matplotlib.pyplot as plt
import easyocr
import numpy as np

from ultralytics import YOLO
import torch

# Global flag to indicate if the loop should exit
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def capture(image):
    cv2.imshow("Captured image", image)
    _, _, width, _ = cv2.getWindowImageRect("Captured image")  # Get the width of the image
    cv2.moveWindow("Captured image", width, 0)
    cv2.imwrite("capture.png", image)
    cv2.waitKey(0)
    return image

def binary_convert(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, im_bw = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite("bi_image.png", im_bw)
    return im_bw

def crop_image(image):
    torch.cuda.set_device(0)
    model = YOLO("license_plate_v8/train/weights/best.pt")
    results = model.predict(image)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cropped_image = image[y1:y2, x1:x2]
            # cv2.imwrite("crop_im.jpg", cropped_image)
    return cropped_image

def filter_image(image):
    # **Apply Highboost Filtering**
    #resize the cropped image
    resized_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # Gaussian blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # Compute mask
    mask = image - blurred
    # Apply highboost filtering
    A = 1  # Boost factor
    highboost_image = image + A * mask
    # Normalize to uint8 for saving
    highboost_image_uint8 = cv2.normalize(highboost_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return highboost_image_uint8

# def segment_image(image):
#     vertical_hist = np.sum(image, axis=0)

def main(cameraId):
    vid = cv2.VideoCapture(cameraId)

    while True:
        _, image = vid.read()
        cv2.imshow("Video Handler", image)
        cv2.moveWindow("Video Handler", 0, 0)
        
        if cv2.waitKey(1) & 0xFF == ord('\r'):
            # capture(image)
            break

    vid.release()               
    cv2.destroyAllWindows()

    # Load the trained YOLO model and move it to the GPU
    cropped_image = None
    cropped_image = crop_image(image)

    if cropped_image is not None:
        bi_pic = binary_convert(cropped_image)
        cv2.imwrite("bi_pic.jpg", bi_pic)
    else:
        print("No license plates detected.")

    # **Apply Highboost Filtering**
    highboost_image_uint8 = filter_image(bi_pic)

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'], gpu=(device.type == 'cuda'))  # Use GPU for EasyOCR

    # Perform OCR on the highboost processed image
    text_results = reader.readtext(highboost_image_uint8, detail=0)  # Extract text only
    print(f'Text extracted from plate image: {text_results}')
    
if __name__ == '__main__':
    cameraId = 0  # Change this to the correct camera ID if needed
    main(cameraId)
