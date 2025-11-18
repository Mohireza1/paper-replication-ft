# import torch
# from torch import nn
# import torch.nn.functional as F
#
#
# class BasicBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, stride=1):
#         super().__init__()
#         self.in_ch = in_ch
#         self.out_ch = out_ch
#         self.stride = stride
#         self.net = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=in_ch,
#                 out_channels=out_ch,
#                 kernel_size=(3, 3),
#                 stride=stride,
#                 padding=1,
#             ),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(),
#             nn.Conv2d(
#                 in_channels=out_ch,
#                 out_channels=out_ch,
#                 kernel_size=(3, 3),
#                 stride=1,
#                 padding=1,
#             ),
#             nn.BatchNorm2d(out_ch),
#         )
#         self.shortcut = nn.Identity()
#         if in_ch != out_ch or stride != 1:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(
#                     self.in_ch,
#                     self.out_ch,
#                     kernel_size=(1, 1),
#                     stride=self.stride,
#                 ),
#                 nn.BatchNorm2d(out_ch),
#             )
#
#     def forward(self, x):
#         out = self.net(x)
#         out = F.relu(out + self.shortcut(x))
#         return out
#
#
# class ResNet(nn.Module):
#     def __init__(self, block_size, in_ch, out_ch, stride=1, num_class=10):
#         super().__init__()
#         self.starter = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=in_ch,
#                 out_channels=out_ch,
#                 kernel_size=(3, 3),
#                 stride=stride,
#                 padding=1,
#             ),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(),
#         )
#         self.layer1 = self.make_layer(block_size, 16, 16, stride=1)
#         self.layer2 = self.make_layer(block_size, 16, 32, stride=2)
#         self.layer3 = self.make_layer(block_size, 32, 64, stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(64, num_class)
#
#         self.net = nn.Sequential(
#             self.starter,
#             self.layer1,
#             self.layer2,
#             self.layer3,
#             self.avgpool,
#             nn.Flatten(),
#             self.fc,
#         )
#
#     def make_layer(self, block_size, in_ch, out_ch, stride):
#         layers = []
#         layers.append(BasicBlock(in_ch, out_ch, stride))
#         for _ in range(block_size - 1):
#             layers.append(BasicBlock(out_ch, out_ch, stride=1))
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         # out = self.starter(x)
#         # out = self.layer1(out)
#         # out = self.layer2(out)
#         # out = self.layer3(out)
#         # out = self.avgpool(out)
#         # out = nn.Flatten(out)
#         # out = self.fc(out)
#         out = self.net(x)
#         return out
#
#
# checkpoint = torch.load("./resnet20-12fca82f.th", map_location="cpu")
# print(list(checkpoint.keys())[:5])
#
# # state_dict = torch.load("./resnet20-12fca82f.th", map_location=torch.device("cpu"))
# # model = ResNet(3, 16, 16)
# # model.load_state_dict(state_dict["state_dict"])
#
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class BasicBlock(nn.Module):
#     def __init__(self, in_planes, planes, stride=1):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#
#         # no learnable downsample in this checkpoint
#         self.in_planes = in_planes
#         self.planes = planes
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#
#         if self.stride != 1 or self.in_planes != self.planes:
#             # spatial downsample via avg pool (no params)
#             identity = F.avg_pool2d(identity, kernel_size=1, stride=self.stride)
#             # channel pad to match planes
#             c_in, c_out = identity.shape[1], self.planes
#             if c_in < c_out:
#                 identity = F.pad(identity, (0, 0, 0, 0, 0, c_out - c_in))
#
#         out = F.relu(out + identity)
#         return out
#
#
# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, num_classes=10):
#         super(ResNet, self).__init__()
#         self.in_planes = 16
#         self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.layer1 = self._make_layer(block, 16, num_blocks[0], 1)
#         self.layer2 = self._make_layer(block, 32, num_blocks[1], 2)
#         self.layer3 = self._make_layer(block, 64, num_blocks[2], 2)
#         self.linear = nn.Linear(64, num_classes)
#
#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for s in strides:
#             layers.append(block(self.in_planes, planes, s))
#             self.in_planes = planes
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = torch.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = nn.functional.avg_pool2d(out, 8)
#         out = torch.flatten(out, 1)
#         out = self.linear(out)
#         return out
#
#
# def resnet20():
#     return ResNet(BasicBlock, [3, 3, 3])
#
#
# import torch
#
# ckpt = torch.load("./resnet20-12fca82f.th", map_location="cpu")
# sd = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
#
# print(
#     "linear.weight shape:", sd["linear.weight"].shape
# )  # expect [10, 64] for CIFAR-10, [100, 64] for CIFAR-100
# print("first conv shape   :", sd["conv1.weight"].shape)  # sanity: [16, 3, 3, 3]
#
# # quick forward-shape sanity: make sure your network outputs the right feature size before the linear
# with torch.no_grad():
#     model = resnet20()
#     model.load_state_dict(sd, strict=True)  # keep it strict to be sure we match exactly
#     model.eval()
#     x = torch.randn(1, 3, 32, 32)
#     feats = model.layer3(
#         model.layer2(model.layer1(torch.relu(model.bn1(model.conv1(x)))))
#     )
#     print("post-layer3 shape:", feats.shape)  # should be [1, 64, 8, 8]
#
# state_dict = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
#
# model = resnet20()
# model.load_state_dict(state_dict)
# model.eval()
#
# x = torch.randn(1, 3, 32, 32)
# print(model(x).shape)
#
# import torchvision
# import torchvision.transforms as transforms
#
# # CIFAR-10 normalization values used during training (match your model's training setup)
# transform = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ]
# )
#
# # Load CIFAR-10 test set
# testset = torchvision.datasets.CIFAR10(
#     root="./data", train=False, download=True, transform=transform
# )
# testloader = torch.utils.data.DataLoader(
#     testset, batch_size=100, shuffle=False, num_workers=2
# )
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# model.eval()
#
# correct = 0
# total = 0
#
# with torch.no_grad():
#     for images, labels in testloader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()
#
# accuracy = 100.0 * correct / total
# print(f"Test Accuracy: {accuracy:.2f}%")
#
#
#

import os
import argparse
from collections import Counter
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T


# --------- CIFAR ResNet20 (Option-A identity shortcut: stride-slice + zero-pad) ---------


class DownsampleA(nn.Module):
    def __init__(self, in_planes, out_planes, stride):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride

    def forward(self, x):
        if self.stride != 1:
            x = x[:, :, :: self.stride, :: self.stride]
        if self.out_planes > self.in_planes:
            pad = self.out_planes - self.in_planes
            x = F.pad(x, (0, 0, 0, 0, 0, pad))
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = (
            nn.Identity()
            if (stride == 1 and in_planes == planes)
            else DownsampleA(in_planes, planes, stride)
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x))
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], 2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, n, stride):
        strides = [stride] + [1] * (n - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.avg_pool2d(x, 8)
        x = torch.flatten(x, 1)
        return self.linear(x)


def resnet20(num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3], num_classes)


# --------- eval helpers ---------


@torch.no_grad()
def eval_once(model, device, norm_name, norm, data_root, batch_size):
    if norm is None:
        tf = T.ToTensor()
    else:
        tf = T.Compose([T.ToTensor(), norm])

    testset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=tf
    )
    loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    model.eval().to(device)

    total = 0
    correct = 0
    hist = Counter()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(
            device, non_blocking=True
        )
        logits = model(imgs)
        pred = logits.argmax(1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
        hist.update(pred.detach().cpu().tolist())

    acc = 100.0 * correct / total
    print(f"[{norm_name}] accuracy = {acc:.2f}%")
    print(f"[{norm_name}] top predicted classes: {hist.most_common(5)}\n")
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="./resnet20-12fca82f.th")
    parser.add_argument("--data", type=str, default="./data")
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()

    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"weights not found: {args.weights}")

    ckpt = torch.load(args.weights, map_location="cpu")
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    if "linear.weight" in sd:
        head_out, head_in = sd["linear.weight"].shape
        if head_out != 10:
            raise RuntimeError(
                f"Checkpoint head is {head_out} classes, not 10. Use CIFAR-100 if head_out=100."
            )

    model = resnet20(num_classes=10)

    # try strict load; fall back to non-strict only if needed (older checkpoints may omit BN num_batches_tracked)
    try:
        missing, unexpected = model.load_state_dict(sd, strict=True)
        if missing or unexpected:
            print(
                "Strict load reported issues. Missing:",
                missing,
                "Unexpected:",
                unexpected,
            )
    except RuntimeError as e:
        print("Strict=True failed, retrying with strict=False.\n", e)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print("Loaded with strict=False. Missing:", missing, "Unexpected:", unexpected)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        x = torch.randn(1, 3, 32, 32).to(device)
        model.to(device).eval()
        z = model.layer3(model.layer2(model.layer1(F.relu(model.bn1(model.conv1(x))))))
        print("feature map after layer3:", tuple(z.shape))  # should be (1, 64, 8, 8)
    # 1) dataloader (use the akamaster normalization that usually pairs with these weights)
    tf = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    testset = torchvision.datasets.CIFAR10(
        "./data", train=False, download=True, transform=tf
    )
    loader = torch.utils.data.DataLoader(
        testset, batch_size=256, shuffle=True, num_workers=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # 2) build confusion matrix: rows=true, cols=pred
    num_classes = 10
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.numpy()
            logits = model(imgs).cpu().numpy()
            preds = logits.argmax(axis=1)
            for t, p in zip(labels, preds):
                cm[t, p] += 1

    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    # 3) find best permutation mapping predicted -> true
    perm = None
    perm_acc = None
    try:
        # Hungarian algorithm (maximize trace by minimizing negative)
        from scipy.optimize import linear_sum_assignment

        cost = cm.max() - cm  # convert to a cost matrix
        r, c = linear_sum_assignment(cost)
        perm = c  # c[j] = predicted class assigned to true class j
        hits = cm[r, c].sum()
        perm_acc = 100.0 * hits / cm.sum()
        print("\n[Hungarian] best permuted accuracy: %.2f%%" % perm_acc)
        print("Mapping true_class -> predicted_class:", perm.tolist())
    except Exception as e:
        print("\nSciPy not available (%s). Using greedy fallback." % type(e).__name__)
        cm_copy = cm.copy()
        perm = [-1] * num_classes
        used_pred = set()
        hits = 0
        for true_cls in range(num_classes):
            # pick the unused predicted class with the highest count for this true class
            best_pred = None
            best_val = -1
            for pred_cls in range(num_classes):
                if pred_cls in used_pred:
                    continue
                if cm_copy[true_cls, pred_cls] > best_val:
                    best_val = cm_copy[true_cls, pred_cls]
                    best_pred = pred_cls
            perm[true_cls] = best_pred
            used_pred.add(best_pred)
            hits += best_val
        perm_acc = 100.0 * hits / cm.sum()
        print("[Greedy] best permuted accuracy: %.2f%%" % perm_acc)
        print("Mapping true_class -> predicted_class:", perm)

    # 4) compare “vanilla” vs remapped predictions on-the-fly
    vanilla_correct = cm.trace()
    vanilla_acc = 100.0 * vanilla_correct / cm.sum()
    print("\nVanilla accuracy (no remap): %.2f%%" % vanilla_acc)
    print("Permutation-aware accuracy:  %.2f%%" % perm_acc)

    # If permuted accuracy is high (~90%+), you can remap predictions at inference time:
    # Example remap function (pred -> remapped_pred):
    inv_map = [None] * num_classes
    for true_cls, pred_cls in enumerate(perm):
        inv_map[pred_cls] = true_cls

    print("\nUse this mapping on model outputs (pred -> CIFAR10 true class):", inv_map)
    # Example:
    # remapped = [inv_map[p] for p in preds]


if __name__ == "__main__":
    main()
