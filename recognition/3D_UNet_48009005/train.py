import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import HipMRI2DDataset, SLICE_DATA_DIR
from modules import UNet2D
from metrics import calculate_dice_iou


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
    
    # Validation dataset
    val_dataset = HipMRI2DDataset(
        SLICE_DATA_DIR / "keras_slices_validate",
        SLICE_DATA_DIR / "keras_slices_seg_validate"
    )

    # Load training data in batches
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    # Load validationdata in batches
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
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
    
    # Keep track of best validation loss
    best_val_loss = float("inf")
            
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
        
        # Validation
        model.eval()
        
        #store validation Dice and IoU values
        dice_scores = {
            class_id: []
            for class_id in range(1,6)
        }
        
        iou_scores = {
            class_id: []
            for class_id in range(1,6)
        }

        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                # forward pass
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss = +- loss.item()
                
                #Choose the predicted class at each pixel
                predictions = torch.argmax(outputs, dim=1)
                
                # Calc Dice and IoU
                results = calculate_dice_iou(predictions, masks, num_classes=6)
                
                for class_id, scores in results.items():
                    dice_scores[class_id].append(scores["dice"])
                    iou_scores[class_id].append(scores["iou"])
        average_val_loss = val_loss / len(val_loader)
        
        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Training Loss: {average_loss:.4f}"
            f"Validation Loss: {average_loss:.4f}"
        )
        
        #display average validation metrics
        for class_id in range(1, 6):
            if dice_scores[class_id]:
                average_dice = sum(dice_scores[class_id]) / len(dice_scores[class_id])
                average_iou = sum(iou_scores[class_id]) / len(iou_scores[class_id])
                
                print(
                    f"Class: {class_id}"
                    f"Dice: {average_dice:.4f}"
                    f"IoU: {average_iou:.4f}"
                )
                
        # Save model with the best validation loss
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            torch.save(model.state_dict(), "best_2d_unet.pth")
            print ("best model saved")
    return model


if __name__ == "__main__":
    train_2d()