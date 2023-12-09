import json
import os
from PIL import Image

def read_classes(classes_file):
    with open(classes_file, "r") as file:
        classes = [line.strip() for line in file.readlines()]
    return classes

def convert_yolov5_to_coco(dataset_dir, output_json_path, split, classes_file):
    coco_data = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    classes_mapping = {class_name: idx + 1 for idx, class_name in enumerate(read_classes(classes_file))}
    image_id = 1

    images_dir = os.path.join(dataset_dir, "images", split)
    labels_dir = os.path.join(dataset_dir, "labels", split)

    for root, dirs, files in os.walk(images_dir):
        for file_name in files:
            if file_name.endswith(".jpg"):
                image_path = os.path.join(root, file_name)
                label_path = os.path.join(labels_dir, os.path.relpath(image_path, images_dir).replace(".jpg", ".txt"))

                # Read image
                image = Image.open(image_path)
                width, height = image.size

                # Generate COCO data for each image
                image_info = {
                    "id": image_id,
                    "file_name": os.path.relpath(image_path, os.path.join(dataset_dir, "images")),
                    "width": width,
                    "height": height,
                }
                coco_data["images"].append(image_info)

                # Read YOLO labels and generate COCO annotations
                with open(label_path, "r") as label_file:
                    yolo_labels = label_file.readlines()

                for line in yolo_labels:
                    class_name, x_center, y_center, box_width, box_height = line.strip().split()
                    category_id = classes_mapping.get(class_name, 1)  # Default to 1 if not found in mapping

                    x_center, y_center, box_width, box_height = map(float, [x_center, y_center, box_width, box_height])
                    x_min = int((x_center - box_width / 2) * width)
                    y_min = int((y_center - box_height / 2) * height)
                    x_max = int((x_center + box_width / 2) * width)
                    y_max = int((y_center + box_height / 2) * height)

                    annotation = {
                        "id": len(coco_data["annotations"]) + 1,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                        "area": (x_max - x_min) * (y_max - y_min),
                        "iscrowd": 0,
                        "segmentation": [],
                    }

                    coco_data["annotations"].append(annotation)

                image_id += 1

    # Add categories based on the mapping
    for class_name, category_id in classes_mapping.items():
        coco_data["categories"].append({
            "id": category_id,
            "name": class_name,
            "supercategory": "object",
        })

    # Save the COCO data to a JSON file
    with open(output_json_path, "w") as json_file:
        json.dump(coco_data, json_file)


def yolo_to_coco(yolo_path):
    root_dataset_path = yolo_path  # Root path to YOLOv5 dataset
    output_coco_json_train = root_dataset_path+"/coco_train.json"
    output_coco_json_val = root_dataset_path+"//coco_val.json"
    classes_file_path = root_dataset_path+"/classes.txt"  # Path to classes.txt file

    convert_yolov5_to_coco(root_dataset_path, output_coco_json_train, "train", classes_file_path)
    convert_yolov5_to_coco(root_dataset_path, output_coco_json_val, "val", classes_file_path)


if __name__ == '__main__':
    yolo_path = r"D:\Python\dates\Drink_284_Detection_Labelme"
    yolo_to_coco(yolo_path)
