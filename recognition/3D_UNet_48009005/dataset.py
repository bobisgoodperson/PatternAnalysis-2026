from pathlib import Path
import random

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

#Dataset folders
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

MR_DIR = DATA_DIR / "semantic_MRs"
LABEL_DIR = DATA_DIR / "semantic_labels_only"
SLICE_DATA_DIR = DATA_DIR / "keras_slices_data"

def find_volume_pairs():
    """
    Find matching HipMRI MRI and segmentation volumes.
    
    Example:
    M030_Week1_LFOV.nii.gz
    M030_Week1_SEMANTIC.nii.gz
    """
    
    pairs = []
    
    for mr_path in sorted(MR_DIR.glob("*_LFOV.nii.gz")):
        
        #create the expected segmentation filename
        label_name = mr_path.name.replace("_LFOV.nii.gz","_SEMANTIC.nii.gz")
        
        label_path = LABEL_DIR / label_name
        
        #patient ID is the first part of the filename
        patient_id = mr_path.name.split("_")[0]
        
        if label_path.exists():
            pairs.append({"patient": patient_id, "mr": mr_path, "label": label_path})
        else:
            print(f"Missing segmentation for: {mr_path.name}")
    return pairs

def split_patients(pairs, seed=67):
    """
    Split HipMRI patients into train, validation and test groups.
    All scans from the same patient remains in the same group.
    """
    
    patients = sorted(set(pair["patient"] for pair in pairs))
    
    random.Random(seed).shuffle(patients)
    
    #38 patients total
    #26 training, 6 validation, 6 testing
    train_patients = patients[:26]
    val_patients = patients[26:32]
    test_patients = patients[32:]
    return train_patients, val_patients, test_patients

def load_data_2D(image_name, normImage=False, dtype=np.float32):
    """
    Load one 2D NIfTI image.

    Adapted from Appendix B of the COMP3710 2026
    Final Assessment Specification v2.4.
    """

    niftiImage = nib.load(image_name)

    inImage = niftiImage.get_fdata(
        caching="unchanged"
    )

    # Sometimes extra dimensions are present in HipMRI data
    if len(inImage.shape) == 3:
        inImage = inImage[:, :, 0]

    inImage = inImage.astype(dtype)

    # Z-score normalisation
    if normImage:
        inImage = (
            inImage - inImage.mean()
        ) / (
            inImage.std() + 1e-8
        )

    return inImage

class HipMRI2DDataset(Dataset):
    """
    Load paired 2D HipMRI MRI slices and segmentation masks.

    NIfTI loading and intensity normalisation are adapted from Appendix B
    of the COMP3710 2026 Assessment Specification
    """

    def __init__(self, image_dir, mask_dir, num_classes=6):
        """
        Store the MRI and segmentation file paths.
        """

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.num_classes = num_classes

        self.image_paths = sorted(
            self.image_dir.glob("case_*.nii.gz")
        )

    def __len__(self):
        """
        Return the number of MRI slices.
        """

        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Load one MRI slice and its matching segmentation mask.
        """

        image_path = self.image_paths[index]

        # Match case_ filename with seg_ filename
        mask_name = image_path.name.replace(
            "case_",
            "seg_",
            1
        )

        mask_path = self.mask_dir / mask_name

        # Load and normalise MRI
        image = load_data_2D(
            image_path,
            normImage=True,
            dtype=np.float32
        )

        # Load segmentation without normalisation
        mask = load_data_2D(
            mask_path,
            normImage=False,
            dtype=np.int64
        )

        # Convert to PyTorch tensors
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask)

        # Convert segmentation mask to one-hot channels
        mask_one_hot = F.one_hot(
            mask,
            num_classes=self.num_classes
        ).permute(2, 0, 1).float()

        return image, mask, mask_one_hot

#check to see if it matches correctly

if __name__ == "__main__":
    pairs = find_volume_pairs()
    print(f"\nMatched MRI/segmentation pairs: {len(pairs)}")
    patients = sorted(set(pair["patient"] for pair in pairs))
    
    print(f"Unique patients: {len(patients)}")
    print(f"Patient IDs: {patients}")
    
    print(f"\nFirst 5 matched pairs:")
    for pair in pairs[:5]:
        print(
            pair["patient"], pair["mr"].name, pair["label"].name
        )
    
    train_patients, val_patients, test_patients = split_patients(pairs)
    
    print("\nPatient Split:")
    print(f"Train: {len(train_patients)}")
    print(f"Validation: {len(val_patients)}")
    print(f"Test: {len(test_patients)}")
    
    print("\n Train Patients:")
    print(train_patients)
    
    print("\n Validation Patients:")
    print(val_patients)
    
    print("\n Test Patients:")
    print(test_patients)
    
    #check that no patient appears in more than one group
    assert set(train_patients).isdisjoint(val_patients)
    assert set(train_patients).isdisjoint(test_patients)
    assert set(val_patients).isdisjoint(test_patients)
    
    print("\nNo patient leakage detected")
    
    print("\nTesting 2D dataset loader...")

    train_2d = HipMRI2DDataset(
        SLICE_DATA_DIR / "keras_slices_train",
        SLICE_DATA_DIR / "keras_slices_seg_train"
    )

    print(f"2D training samples: {len(train_2d)}")

    image, mask, mask_one_hot = train_2d[0]

    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"One-hot shape: {mask_one_hot.shape}")

    print(f"Image mean: {image.mean().item():.4f}")
    print(f"Image std: {image.std().item():.4f}")
    print(f"Mask labels: {mask.unique()}")