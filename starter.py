import torch

checkpoint = torch.load("resnet20-12fca82f.th", map_location="cpu")
state_dict = checkpoint["state_dict"]

for name, tensor in state_dict.items():
    print(f"{name:40s} {tuple(tensor.shape)}")
