import time

import torch
from torch.utils.data import DataLoader

from dataset import *
from modules import UNet2D, UNet3D
from metrics import calculate_dice_iou

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def evaluate_2d(model_path=None):
    """
    Evaluate the trained 2D U-Net on the held-out HipMRI test set.

    Reports per-class Dice and IoU, average inference time per slice,
    and peak GPU memory usage.
    """

    # Use the saved best model by default
    if model_path is None:
        model_path = PROJECT_DIR / "best_2d_unet.pth"

    # Use GPU if available
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # Load the held-out test dataset
    test_dataset = HipMRI2DDataset(
        SLICE_DATA_DIR / "keras_slices_test",
        SLICE_DATA_DIR / "keras_slices_seg_test"
    )

    # Batch size 1 is required because slice sizes vary
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False
    )

    print(f"Test slices: {len(test_dataset)}")

    # Create the same model architecture used during training
    model = UNet2D(
        in_channels=1,
        num_classes=6
    ).to(device)

    # Load the trained model weights
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model.eval()

    # Store Dice and IoU values for each class
    dice_scores = {
        class_id: []
        for class_id in range(1, 6)
    }

    iou_scores = {
        class_id: []
        for class_id in range(1, 6)
    }

    inference_times = []

    # Warm up the GPU before timing
    if len(test_dataset) > 0:
        image, _, _ = test_dataset[0]
        image = image.unsqueeze(0).to(device)

        with torch.no_grad():
            model(image)

        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

    # Evaluate without calculating gradients
    with torch.no_grad():

        for images, masks, _ in test_loader:

            images = images.to(device)
            masks = masks.to(device)

            # Wait for previous GPU operations before timing
            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            # Forward pass
            outputs = model(images)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            inference_times.append(
                end_time - start_time
            )

            # Choose the most likely class at each pixel
            predictions = torch.argmax(
                outputs,
                dim=1
            )

            # Calculate Dice and IoU
            results = calculate_dice_iou(
                predictions,
                masks,
                num_classes=6
            )

            for class_id, scores in results.items():
                dice_scores[class_id].append(
                    scores["dice"]
                )

                iou_scores[class_id].append(
                    scores["iou"]
                )

    print("\n2D U-Net Test Results")

    class_dice_averages = []
    class_iou_averages = []

    # Display average result for each anatomical class
    for class_id in range(1, 6):

        if dice_scores[class_id]:

            average_dice = (
                sum(dice_scores[class_id])
                / len(dice_scores[class_id])
            )

            average_iou = (
                sum(iou_scores[class_id])
                / len(iou_scores[class_id])
            )

            class_dice_averages.append(average_dice)
            class_iou_averages.append(average_iou)

            print(
                f"Class {class_id}: "
                f"Dice = {average_dice:.4f}, "
                f"IoU = {average_iou:.4f}"
            )

    # Mean across foreground classes
    if class_dice_averages:

        mean_dice = (
            sum(class_dice_averages)
            / len(class_dice_averages)
        )

        mean_iou = (
            sum(class_iou_averages)
            / len(class_iou_averages)
        )

        print(f"\nMean Dice: {mean_dice:.4f}")
        print(f"Mean IoU: {mean_iou:.4f}")

    # Average inference time for one 2D slice
    if inference_times:

        average_time = (
            sum(inference_times)
            / len(inference_times)
        )

        print(
            f"Average inference time per slice: "
            f"{average_time:.6f} seconds"
        )

    # Report peak GPU memory
    if device.type == "cuda":

        peak_memory = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

        print(
            f"Peak GPU memory: "
            f"{peak_memory:.3f} GB"
        )


def visualise_3d():
    """
    Visualise the weighted 3D U-Net prediction
    using a validation MRI volume.
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load the saved weighted model
    model = UNet3D(in_channels=1, num_classes=6).to(device)

    checkpoint = PROJECT_DIR / "best_3d_unet_weighted.pth"

    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )

    model.eval()

    # Use validation patients, not the held-out test patients
    pairs = find_volume_pairs()
    _, val_patients, _ = split_patients(pairs)

    val_pairs = [pair for pair in pairs if pair["patient"] in val_patients]

    dataset = HipMRI3DDataset(val_pairs)

    # Find a validation slice containing Class 5
    selected = None

    for index in range(len(dataset)):
        image, mask, _ = dataset[index]

        class_counts = (mask == 5).sum(dim=(1, 2))

        if class_counts.max().item() > 0:
            slice_index = class_counts.argmax().item()
            selected = (image, mask, val_pairs[index])
            break

    if selected is None:
        raise ValueError("No Class 5 validation slice found")

    image, mask, pair = selected

    # Predict the segmentation for the full 3D volume
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        prediction = output.argmax(dim=1).squeeze(0).cpu()

    # Show the same slice across all three images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(
        image[0, slice_index].numpy(),
        cmap="gray"
    )
    axes[0].set_title("MRI Image")

    axes[1].imshow(
        mask[slice_index].numpy(),
        cmap="tab10",
        vmin=-0.5,
        vmax=5.5
    )
    axes[1].set_title("Ground Truth")

    seg_image = axes[2].imshow(
        prediction[slice_index].numpy(),
        cmap="tab10",
        vmin=-0.5,
        vmax=5.5
    )
    axes[2].set_title("3D U-Net Prediction")

    for ax in axes:
        ax.axis("off")

    fig.colorbar(
        seg_image,
        ax=axes[1:],
        ticks=range(6),
        shrink=0.75
    )

    fig.suptitle(
        f"{pair['mr'].name} - Slice {slice_index}"
    )

    # Save the visualisation
    output_dir = PROJECT_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "3d_validation_prediction.png"

    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Visualisation saved: {output_path}")

def visualise_3d_overlay():
    """
    Visualise the weighted 3D U-Net prediction as an overlay
    on top of the MRI image using a validation MRI volume.
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load the saved weighted 3D U-Net
    model = UNet3D(
        in_channels=1,
        num_classes=6
    ).to(device)

    checkpoint = PROJECT_DIR / "best_3d_unet_weighted.pth"

    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    # Use validation patients only
    pairs = find_volume_pairs()
    _, val_patients, _ = split_patients(pairs)

    val_pairs = [
        pair for pair in pairs
        if pair["patient"] in val_patients
    ]

    dataset = HipMRI3DDataset(val_pairs)

    # Choose a validation volume with Class 5 present
    selected = None

    for index in range(len(dataset)):
        image, mask, _ = dataset[index]

        class5_counts = (mask == 5).sum(dim=(1, 2))

        if class5_counts.max().item() > 0:
            slice_index = class5_counts.argmax().item()
            selected = (image, mask, val_pairs[index], slice_index)
            break

    if selected is None:
        raise ValueError("No validation slice containing Class 5 was found")

    image, mask, pair, slice_index = selected

    # Predict segmentation
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        prediction = torch.argmax(output, dim=1).squeeze(0).cpu()

    # MRI slice
    mri_slice = image[0, slice_index].numpy()
    mask_slice = mask[slice_index].numpy()
    pred_slice = prediction[slice_index].numpy()

    # Make label 0 transparent, colours for labels 1-5
    overlay_cmap = ListedColormap([
        (0, 0, 0, 0),      # class 0 transparent
        (0, 1, 0, 0.45),   # class 1 green
        (0.6, 0.4, 1, 0.45), # class 2 purple
        (1, 0.4, 0.8, 0.45), # class 3 pink
        (0.7, 0.7, 0.7, 0.45), # class 4 grey
        (0, 1, 1, 0.45)    # class 5 cyan
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Ground truth overlay
    axes[0].imshow(mri_slice, cmap="gray")
    axes[0].imshow(
        mask_slice,
        cmap=overlay_cmap,
        vmin=0,
        vmax=5,
        interpolation="nearest"
    )
    axes[0].set_title("MRI + Ground Truth")
    axes[0].axis("off")

    # Prediction overlay
    axes[1].imshow(mri_slice, cmap="gray")
    axes[1].imshow(
        pred_slice,
        cmap=overlay_cmap,
        vmin=0,
        vmax=5,
        interpolation="nearest"
    )
    axes[1].set_title("MRI + 3D U-Net Prediction")
    axes[1].axis("off")

    fig.suptitle(
        f"{pair['mr'].name} - Slice {slice_index}"
    )

    output_dir = PROJECT_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "3d_validation_overlay.png"

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )
    plt.close(fig)

    print(f"Overlay visualisation saved: {output_path}")
if __name__ == "__main__":
    evaluate_2d()