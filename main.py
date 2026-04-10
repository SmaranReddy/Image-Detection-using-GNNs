import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

from models.hybrid_model import CNNViTGNN
from training.train import train


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== Transforms =====
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.247, 0.243, 0.261]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.247, 0.243, 0.261]
        )
    ])

    # ===== Dataset =====
    train_dataset = CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform
    )

    val_dataset = CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=val_transform
    )

    # ===== DataLoader =====
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )

    # ===== SELECT MODE HERE =====
    # Options:
    # "cnn"
    # "cnn_vit"
    # "cnn_vit_gnn"

    model = CNNViTGNN(
        num_classes=10,
        mode="cnn"   # 👈 CHANGE THIS FOR EXPERIMENTS
    )

    # ===== Train =====
    train(
        model,
        train_loader,
        val_loader,
        epochs=10,      # start small for testing
        lr=1e-3,
        device=device
    )


if __name__ == "__main__":
    main()