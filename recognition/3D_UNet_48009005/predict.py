import time

import torch
from torch.utils.data import DataLoader

from dataset import HipMRI2DDataset, SLICE_DATA_DIR, PROJECT_DIR
from modules import UNet2D
from metrics import calculate_dice_iou


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


if __name__ == "__main__":
    evaluate_2d()