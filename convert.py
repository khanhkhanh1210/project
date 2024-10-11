import os
import xml.etree.ElementTree as ET

def convert_xml_to_yolo(xml_file, output_path):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_width = int(root.find('size/width').text)
    image_height = int(root.find('size/height').text)

    for obj in root.findall('object'):
        class_name = obj.find('name').text
        # Map class_name to YOLO class index
        class_index = 0 if class_name == "licence" else -1
        
        # Extract bounding box
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)

        # Calculate YOLO format values
        x_center = (xmin + xmax) / 2.0 / image_width
        y_center = (ymin + ymax) / 2.0 / image_height
        box_width = (xmax - xmin) / image_width
        box_height = (ymax - ymin) / image_height

        # Create YOLO format string
        yolo_format = f"{class_index} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n"

        # Write to corresponding YOLO .txt file
        txt_filename = os.path.join(output_path, os.path.basename(xml_file).replace('.xml', '.txt'))
        with open(txt_filename, 'a') as f:
            f.write(yolo_format)

if __name__=='__main__':
    #delete existing files in 3 folder test, val, train
    for folder in os.listdir('datasets/labels'):
        for file in os.listdir(f'datasets/labels/{folder}'):
            os.remove(f'datasets/labels/{folder}/{file}')
    
    # Get all XML files
    for i in range(0, len(os.listdir('annotations'))):
        if i < 320:
            convert_xml_to_yolo(f'annotations/Cars{i}.xml', 'datasets/labels/train')
        elif i < 360:
            convert_xml_to_yolo(f'annotations/Cars{i}.xml', 'datasets/labels/test')
        else:
            convert_xml_to_yolo(f'annotations/Cars{i}.xml', 'datasets/labels/val')
        
    print("Conversion completed: Training, validation, and test sets prepared.")