''' SCRIPT FOR TRAINING WITH TORCHVISION PRETRAINED MODELS

Comparison from copilot:

Model                     | Speed     | Accuracy  | Notes
---------------------------------------------------------------------------------
fasterrcnn_resnet50_fpn   | Medium    | High      | Strong baseline
retinanet_resnet50_fpn    | Fast      | Medium    | One-stage detector
ssdlite320_mobilenet_v3   | Very fast | Lower     | Good for mobile/edge
maskrcnn_resnet50_fpn     | Medium    | High      | Adds instance segmentation

'''

# imports
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
# from custom_dataset import OffroadDataset
import torch

# Load pretrained model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Replace head
num_classes = 5 
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load dataset
train_dataset = OffroadDataset('offroad_dataset/images/train', 'offroad_dataset/labels/train', transforms=...)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Train loop (simplified)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

for epoch in range(10):
    model.train()
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

