# Change Detection with Siamese Network

A PyTorch implementation of a Siamese Network for change detection using the [LEVIR-CD dataset](https://www.kaggle.com/datasets/mdrifaturrahman33/levir-cd). The project detects changes between pairs of satellite images by classifying them as "changed" or "unchanged."

## Features
- **Siamese Network**: Uses ResNet18 backbone to compare image pairs.
- **Contrastive Loss**: Optimizes the model to distinguish between similar and different image pairs.
- **LEVIR-CD Dataset**: Processes satellite image pairs and labels for training, validation, and testing.
- **Training & Inference**: Includes scripts for model training and visualization of predictions.

## Sample Inference Results
- **Change Example**: A pair of images showing detected changes.
  ![Change Example](https://github.com/uyenvoaero/change_detection/blob/main/output/test_10.png)
- **Unchange Example**: A pair of images showing no detected changes.
  ![Unchange Example](https://github.com/uyenvoaero/change_detection/blob/main/output/test_65.png)

## Output
- **Models**: Saved in `./output/expN/models/` (best and last checkpoints).
- **Graphs**: Training metrics (loss and accuracy) saved in `./output/expN/graphs/`.
- **Inference Results**: Visualizations of correct and wrong predictions saved in `./output/expN/inference_results/`.
