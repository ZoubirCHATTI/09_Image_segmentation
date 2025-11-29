# Brain Mask Segmentation with Deep Learning (U-Net)

## Project Description

This project implements a **U-Net** convolutional neural network for the binary segmentation of brain masks from T1-weighted (T1w) MRI images. The main goal is to evaluate the performance of a "basic" U-Net and to demonstrate the impact of **Data Augmentation** on the model's robustness and accuracy.

The code is structured to be executed in a Jupyter/Colab-like environment, integrating data download, model definition, training, and evaluation.

## Key Features

*   **U-Net Architecture:** Implementation of a standard U-Net for semantic segmentation.
*   **NIfTI Data Loading:** Use of the `nibabel` library to read 3D volumes in `.nii.gz` format and process them as 2D slices.
*   **Data Augmentation:** Application of geometric transformations (rotation, shift, flip) via `ImageDataGenerator` to enrich the training set and improve model generalization.
*   **Evaluation Metrics:** The model is evaluated using standard segmentation metrics:
    *   **Dice Coefficient** (Dice Score)
    *   **Hausdorff Distance**
*   **Visualization:** Display of learning curves (loss and accuracy) and segmentation results.

## Prerequisites

The project requires the following Python libraries. They can be installed via `pip`:

| Library | Primary Role |
| :--- | :--- |
| `tensorflow` / `keras` | U-Net model definition and training. |
| `numpy` | Numerical data manipulation. |
| `matplotlib` | Visualization of images and learning curves. |
| `nibabel` | Reading and writing NIfTI medical image files (`.nii.gz`). |
| `scikit-image` (`skimage`) | Image processing functions (resizing). |
| `scipy` | Calculation of the Hausdorff Distance. |
| `gdown` | Downloading data archives from Google Drive. |

## Installation and Configuration

1.  **Install Python Dependencies:**
    ```bash
    pip install tensorflow numpy matplotlib nibabel scikit-image scipy gdown
    ```

2.  **Download Data:**
    The Python script includes commands to automatically download and unzip the training (`training`) and testing (`testing`) datasets from Google Drive using `gdown`.

    ```python
    !pip install gdown
    !gdown https://drive.google.com/uc?id=17gb7VFUgoHzUWc3eH_gxdmNekLLZQEIO # training.zip
    !gdown https://drive.google.com/uc?id=16-bZCe4whMnMfBjXBZwhPS16Uxet_x07 # testing.zip
    !unzip training
    !unzip testing
    ```

## Usage

The project is contained within a single Python script (`tp_brainmask_seg_with_unet_en_clean.py`).

1.  **Run the Script:**
    The script is designed to be executed sequentially. If you are using an environment like Colab or Jupyter Notebook, simply run all cells.
    If running from the command line, ensure all data download and unzipping steps are complete and that the `/content/training` and `/content/testing` directories exist and contain the data.

    ```bash
    python brainmask_seg_with_unet.ipynb
    ```

2.  **Script Steps:**
    *   **Data Loading:** MRI slices and masks are loaded and prepared.
    *   **Model Definition:** The U-Net model is created.
    *   **Training:** The model is trained with and without data augmentation (depending on the script configuration).
    *   **Evaluation:** Predictions are made on the test set, and metrics (Dice, Hausdorff) are calculated and displayed.

## U-Net Architecture

The implemented U-Net model is a classic semantic segmentation architecture with skip connections to combine low-level and high-level features.

| Block | Image Size | Number of Filters | Operations |
| :--- | :--- | :--- | :--- |
| **Contraction (Encoder)** | 128x128 to 8x8 | 16, 32, 64, 128, 256 | `Conv2D` (ReLU), `Dropout`, `MaxPooling2D` |
| **Expansion (Decoder)** | 8x8 to 128x128 | 128, 64, 32, 16 | `Conv2DTranspose`, `Concatenate` (Skip Connection), `Conv2D` (ReLU), `Dropout` |
| **Output** | 128x128 | 1 | `Conv2D` (Sigmoid) |

## Metrics

The script uses two main metrics to evaluate the quality of the segmentation:

### 1. Dice Coefficient (Dice Score)

The Dice Coefficient is a measure of overlap between the prediction and the ground truth. It is defined as:

$$
\text{Dice} = \frac{2 \cdot |A \cap B|}{|A| + |B|}
$$

Where $A$ is the predicted mask and $B$ is the ground truth mask. A value close to 1 indicates perfect segmentation.

### 2. Hausdorff Distance

The Hausdorff Distance measures the maximum distance between two sets of points (the mask boundaries). A lower value indicates that the predicted boundaries are closer to the actual boundaries.

## Author

**Manus AI**
(Based on a brain mask segmentation workshop)
