from ultralytics import YOLO

def run_inference(image_path, output_file):
    # Load YOLOv8 model
    model = YOLO('license_plate/train/weights/best.pt')
    # Run inference on the image
    results = model(image_path)
    # Write results to a text file
    with open(output_file, 'w') as file:
        file.write(str(results))

if __name__ == '__main__':
    image_path = 'datasets/images/test/Cars320.png'  
    output_file = 'results.txt'
    run_inference(image_path, output_file)