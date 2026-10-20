# Feasibility Review
## User Need, Scope and Acceptance Criteria
Accurate segmentation of prostate MRI volumes is important in radiotherapy planning, where anatomical structure must be delineated reliably while minimising errors near clinically important boundaries. The purpose of this prototype is to investigate whether incorporating volumetric spatial information using a 3D Unet peovides more consistent prostate segmentation than a conventional 2D Unet baseline, particulatly at difficult anatomical boundaries and across adjacent MRI slices. With the core investigation being whether true 3D volumetric continuity reduces hazardous boundary and slice-transition errors compared with a standard 2D baseline, and whether any improvement justifies the additional GPU memory and training time required by volumetric processing.
The following acceptance criteria will be used:
1. The Final 3D Unet will target a Dice score of at least 0.70 on the HipMRI data set
2. No patient appears in more than one of train/validation/test do this by shuffling and seperating the patients into its specific section ie Patient 1 is dedicated to training
3. The 2D Unet baseline and 3D unet will be evaluated on the same heId out patients using dice coefficient and Intersection Over Union (IoU)
4. All representative segmentation failures will be investigated, with particular attention to difficult anatomical boundaries and slice transition failures.
5. Peak GPU VRAM and per case inference time will be measured to assess whether any segmentation improvement obtained from 3D processing justifies the additional computational cost. 
## Model Choice and Course Concepts
A standard 2D Unet will be used as the baseline model, while the proposed final model us a 3D Unet adapted to the HipMRI volumetric dataset.
A previously developed 2D Unet from DEMO 2 will be adapted for the baseline. Reusable components include the encoder-decoder structure, skip connections, training and validation framework, dice-based evaluation and segmentation visualisation. The hipMRI data-loading pipeline, patient level splitting and volumetric 3D Unet implementation will be developed specifically for this project.
The final architecture will be based on the 3D Unet approach of the paper given. The 3D Unet extends the Unet concept to volumetric data, allowing convolutional features to be learned across the depth dimension as well as the two in plane spatial dimensions. The architecture therefore uses volumetric convolutions and skip connections to reattain spatial detain while incorporating context from neighbouring MRI slices.
There is no lecture content for 3D Unets, only 2D ones, Thus the paper will be the main source of architectural details
## Preliminary Feasibility Evidence
An initial audit was successfully completed using the paired HipMRI volume:
M030_Week1_LFOV.nii.gz and M030_Week1_SEMANTIC.nii.gz
Both the MRI and segmentation had dimensions of: 256 × 256 × 128 voxels
and matching voxel spacing of approximately: 1.680 × 1.680 × 1.560 mm
The MRI and segmentation had both matching spatial dimensions and matching affine matrices, indicating correct spatial alignment for the inspected case.
Z-score intensity normalisation was successfully applied to the MRI and produced:
Mean = 0.0000
Standard deviation = 1.0000

Six segmentation values were observed, labelled 0–5. Their distributions were:
| Label | Percentage of Voxels (%) |
| - | - |
| 0 | 61.072 |
| 1 | 35.112 |
| 2 | 2.885 |
| 3 | 0.758 |
| 4 | 0.102 |
| 5 | 0.070 |

Label 0 will be excluded for future testing since it is just the background.
This demonstrates substantial class imbalance, with the two smallest labelled regions occupying only approximately 0.10% and 0.07% of the volume. This reinforces the requirement for class-specific Dice and IoU rather than overall voxel accuracy.
The anatomical mapping of label values 1–5 has not yet been assumed and will be verified before model training. The HipMRI dataset itself contains annotations for the body outline, bone, bladder, rectum and prostate. 
The successful loading, normalisation and spatial-alignment test indicates that the dataset can be processed successfully for the proposed 3D segmentation pipeline.

## Risks, Computing Budget and Fallback
The primary technical risk is GPU memory usage, as full 256 × 256 × 128 volumes processed using 3D convolutions require substantially more memory than 2D slices. Additional risks include severe class imbalance, loss of small anatomical structures during downsampling, inconsistent label encoding across cases, and patient-level data leakage.
Training and GPU testing will be performed on the Rangpur comp3710 GPU partition using a single A100 GPU. Initial experiments will use short smoke tests and a batch size of 1 before longer training jobs are submitted. Peak VRAM will be measured during these tests.
The next experiment will audit multiple HipMRI cases to confirm label consistency and patient identifiers, implement the patient-level split, and then perform a single 3D U-Net forward-pass smoke test. Where the smoke test is just loading and testing a small test point to make sure that the model works.
If full-volume training exceeds the available GPU memory, the fallback will be to reduce the input volume dimensions or use 3D subvolumes/patches with a small batch size while retaining the required 3D U-Net architecture. Mixed-precision training may also be used if required. The 2D U-Net will remain the comparison baseline rather than replacing the final 3D model.
