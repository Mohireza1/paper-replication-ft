from numpy import mod
import torch
from torch.utils.data import TensorDataset, DataLoader

# from torch import nn
import torchvision
from torchvision.models import MobileNet_V2_Weights

weights = torch.load("./weights.pkl", map_location="cpu")

# print(weights)
model = torchvision.models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.last_channel, 10)
model.load_state_dict(weights)


transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

data = torchvision.datasets.CIFAR10(root="./data/", train=False, transform=transform)
datald = DataLoader(
    data,
    batch_size=128,
    shuffle=False,
)


model.eval()
answers = []
with torch.no_grad():
    for d in datald:
        pred = model.forward(d[0])
        print(pred)
