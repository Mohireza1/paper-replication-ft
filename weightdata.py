import torch

data = torch.load("weights.pkl", map_location="cpu")

for point in data:
    print(point)
