import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import HipMRI2DDataset, SLICE_DATA_DIR
from modules import UNet2D


def train_2d(epochs=5, batch_size=4, learning_rate=0.001):
    """
    Train the 2D U-Net baseline on the HipMRI training slices.
    """

    # Use GPU if available
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Training dataset
    train_dataset = HipMRI2DDataset(
        SLICE_DATA_DIR / "keras_slices_train",
        SLICE_DATA_DIR / "keras_slices_seg_train"
    )

    # Load data in batches
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Create model
    model = UNet2D(
        in_channels=1,
        num_classes=6
    ).to(device)

    # Multi-class segmentation loss
    criterion = nn.CrossEntropyLoss()

    # Optimiser
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # Training epochs
    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        for images, masks, mask_one_hot in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}"
        ):

            images = images.to(device)
            masks = masks.to(device)

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, masks)

            # Backpropagation
            loss.backward()

            # Update model weights
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Training Loss: {average_loss:.4f}"
        )

    return model


if __name__ == "__main__":
    train_2d()