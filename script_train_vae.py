""" Train variational auto-encoder. """

import torch
import torchvision
from torch import optim
from torch.utils.data import DataLoader

from reference_models.vae import VariationalAutoEncoder

torch.manual_seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

for dataset_name, raw_dataset in zip(["mnist", "fmnist", "kmnist"], [
    torchvision.datasets.MNIST(
        root="./data/image-data", train=True,
        transform=torchvision.transforms.ToTensor(), download=True,
    ),
    torchvision.datasets.FashionMNIST(
        root="./data/image-data", train=True,
        transform=torchvision.transforms.ToTensor(), download=True,
    ),
    torchvision.datasets.KMNIST(
        root="./data/image-data", train=True,
        transform=torchvision.transforms.ToTensor(), download=True,
    ),
]):
    # Prepare data
    trainloader = DataLoader(raw_dataset, batch_size=64, shuffle=True)

    # Create model
    model = VariationalAutoEncoder()
    model.to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters())

    # Training loop
    for epoch in range(1000):
        running_loss = 0.0
        loss_count = 0
        for data in trainloader:
            # Forward pass
            inputs, _ = data
            inputs = inputs.reshape(inputs.shape[0], -1)
            inputs = inputs.to(device)
            loss, _, _ = model(inputs)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loss_count += 1

        print(f"[{epoch + 1: >4}] {running_loss / loss_count:.6f}")
        running_loss = 0.0
        loss_count = 0

    # Save model
    model.eval()
    torch.save(model.state_dict(), f"./saved_models/vae-{dataset_name}.pt")
