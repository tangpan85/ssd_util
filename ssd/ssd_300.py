import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt

# Step 1: 加载预训练模型
model = ssdlite320_mobilenet_v3_large(pretrained=True)
model.eval()


# Step 2: 数据预处理
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0)
    return image_tensor


# Step 3: 进行推理
def infer(image_path):
    image_tensor = preprocess_image(image_path)

    with torch.no_grad():
        predictions = model(image_tensor)

    return predictions


# Step 4: 可视化结果
def visualize(image_path, predictions, threshold=0.5):
    image = Image.open(image_path).convert("RGB")

    # Filter predictions based on confidence threshold
    boxes = predictions[0]['boxes'][predictions[0]['scores'] > threshold]
    labels = predictions[0]['labels'][predictions[0]['scores'] > threshold]

    # Visualize the predictions
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    for box, label in zip(boxes, labels):
        box = [round(coord.item(), 2) for coord in box]
        box = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
        rect = plt.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(box[0], box[1], f'Label: {label.item()}', color='r', backgroundcolor='w', alpha=0.7)

    plt.axis('off')
    plt.show()


# Example Usage
image_path = "1.jpg"
predictions = infer(image_path)
visualize(image_path, predictions)
