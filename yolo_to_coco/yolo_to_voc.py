import os
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom
from PIL import Image

def create_xml(image_name, labels, image_width, image_height):
    root = Element('annotation')
    SubElement(root, 'folder').text = 'JPEGImages'
    SubElement(root, 'filename').text = image_name

    size = SubElement(root, 'size')
    SubElement(size, 'width').text = str(image_width)
    SubElement(size, 'height').text = str(image_height)
    SubElement(size, 'depth').text = '3'  # RGB 图像深度

    for label in labels:
        obj = SubElement(root, 'object')
        SubElement(obj, 'name').text = str(label[0])
        SubElement(obj, 'pose').text = 'Unspecified'
        SubElement(obj, 'truncated').text = '0'
        SubElement(obj, 'difficult').text = '0'

        bbox = SubElement(obj, 'bndbox')
        xmin = int(label[1] * image_width)
        ymin = int(label[2] * image_height)
        xmax = int((label[1] + label[3]) * image_width)
        ymax = int((label[2] + label[4]) * image_height)

        SubElement(bbox, 'xmin').text = str(xmin)
        SubElement(bbox, 'ymin').text = str(ymin)
        SubElement(bbox, 'xmax').text = str(xmax)
        SubElement(bbox, 'ymax').text = str(ymax)

    return root

def prettify(elem):
    rough_string = tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def get_image_size(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height

def yolo_to_voc_recursive(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for label_file in files:
            if label_file.endswith('.txt'):
                image_name = label_file.replace('.txt', '.jpg')
                image_path = os.path.join(root, image_name)
                label_path = os.path.join(root, label_file)

                # 读取图像尺寸
                image_width, image_height = get_image_size(image_path)

                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    labels = [list(map(float, line.strip().split())) for line in lines]

                xml_root = create_xml(image_name, labels, image_width, image_height)
                xml_string = prettify(xml_root)

                relative_path = os.path.relpath(label_path, input_dir)
                output_path = os.path.join(output_dir, relative_path.replace('.txt', '.xml'))
                with open(output_path, 'w') as xml_file:
                    xml_file.write(xml_string)


def yolo_to_voc(yolo_path):

    yolo_to_voc_recursive(yolo_path+'labels/train', yolo_path+'/Annotations/train')
    yolo_to_voc_recursive(yolo_path+'path/to/yolo/labels/val', yolo_path+'/Annotations/val')

if __name__ == '__main__':
    yolo_path = r"D:\Python\dates\Drink_284_Detection_Labelme"
    yolo_to_voc(yolo_path)