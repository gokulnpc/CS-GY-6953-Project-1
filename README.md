# CIFAR-10 Image Classification with Custom ResNet

This repository contains the implementation of a custom ResNet architecture for CIFAR-10 image classification, developed as part of the CS-GY 6953 Deep Learning course at NYU.

## Project Overview

We implement a custom ResNet architecture that achieves competitive performance on the CIFAR-10 dataset while maintaining a relatively small parameter count. The model incorporates several modern deep learning techniques:

- Residual connections for better gradient flow
- Advanced data augmentation (Mixup and CutMix)
- Test-time augmentation (TTA) with model ensembling

## Model Architecture

Our custom ResNet architecture features:

- Initial convolution layer with 84 channels
- Three residual stages with increasing channel dimensions (84→168→336)
- Each stage contains 2 residual blocks with batch normalization
- Global average pooling and dropout (p=0.5) for regularization
- Final fully-connected layer for 10-class classification

Total trainable parameters: ~4.7M

## Training Strategy

We employ several modern training techniques:

- **Data Augmentation:**

  - Random crop
  - Horizontal flip
  - Random erasing
  - Mixup and CutMix (50% probability each)

- **Training Configuration:**
  - Label smoothing (0.1) with cross-entropy loss
  - Adam optimizer
  - Learning rate: 1e-3
  - Weight decay: 1e-4
  - Cosine annealing learning rate schedule
  - Early stopping with patience of 25 epochs
  - Total epochs: 250

## Results

Our model achieves competitive performance on CIFAR-10:

| Model     | Parameters (M) | Validation Accuracy (%) |
| --------- | -------------- | ----------------------- |
| Ours      | 4.7            | 93.68                   |
| ResNet-18 | 11.2           | 93.0                    |
| DenseNet  | 25.6           | 93.8                    |

- Maximum training accuracy: 98%
- Best validation accuracy: 93.68%
- Training time: ~2 hours on Google Colab T4 GPU

## Repository Structure

- `resnet.py`: Implementation of the custom ResNet architecture and training logic
- `predict.py`: Inference code with test-time augmentation
- `checkpoint.pth`: Trained model weights

## Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib
- tqdm

## Usage

1. Training the model:

```bash
python resnet.py
```

2. Making predictions:

```bash
python predict.py
```

## Citation

If you find this work useful, please cite:

```bibtex
@misc{narayanan2024cifar,
  author = {Narayanan, Gokuleshwaran and Jammalamadugu, Solomon Martin},
  title = {CIFAR-10 Image Classification with Custom ResNet},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/gokulnpc/CS-GY-6953-Project-1}
}
```

## Acknowledgments

We thank the course staff of CS-GY 6953 at NYU for their guidance and feedback throughout this project.
