from pathlib import Path
import random

#Dataset folders
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

MR_DIR = DATA_DIR / "semantic_MRs"
LABEL_DIR = DATA_DIR / "semantic_labels_only"

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