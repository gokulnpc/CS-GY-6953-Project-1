import os 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from cifar_resnet import CustomResNet
import multiprocessing

multiprocessing.set_start_method('spawn', force=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

def prepare_test_data(data):
    """
    data: A NumPy array of shape (N, 32, 32, 3).
    Returns: A torch.FloatTensor of shape (N, 3, 32, 32), normalized.
    """
    # Convert to float32 if not already
    data = data.astype(np.float32)
    # Transpose from (N, H, W, C) to (N, C, H, W)
    data = np.transpose(data, (0, 3, 1, 2))  # shape becomes (N, 3, 32, 32)
    # Scale to [0, 1]
    data /= 255.0
    # CIFAR-10 mean/std
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32).reshape(1, 3, 1, 1)
    data = (data - mean) / std  # broadcast normalization
    return torch.from_numpy(data)

def visualize_samples(images, num_samples=5):
    """
    Visualize a few sample images from the preprocessed tensor.
    `images` is a tensor of shape (N, 3, 32, 32).
    """
    # Undo normalization for display purposes
    images = images.clone().numpy()
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3,1,1)
    std = np.array([0.2023, 0.1994, 0.2010]).reshape(3,1,1)
    images = images * std + mean  # reverse normalization
    # Clip pixel values to [0,1]
    images = np.clip(images, 0, 1)
    
    plt.figure(figsize=(num_samples*2, 2))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        # Convert from (C, H, W) to (H, W, C)
        plt.imshow(np.transpose(images[i], (1, 2, 0)))
        plt.axis('off')
    plt.suptitle("Sample Preprocessed Custom Test Images")
    plt.show()

def make_predictions():
    # Load the model
    model = CustomResNet().to(device)
    checkpoint = torch.load('checkpoint.pth', map_location=device)
    # Check if the checkpoint contains a dictionary with model_state_dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')} with best validation accuracy: {checkpoint.get('best_val_acc', 'unknown')}%")
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Load and prepare test data
    test_data = unpickle('cifar_test_nolabel.pkl')
    test_images = prepare_test_data(test_data[b'data'])
    
    # Visualize a few samples before inference
    visualize_samples(test_images, num_samples=5)
    
    # Make predictions in batches
    predictions = []
    batch_size = 100
    with torch.no_grad():
        for i in range(0, test_images.size(0), batch_size):
            batch = test_images[i:i+batch_size].to(device)
            outputs = model(batch)
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy().tolist())
    
    # Convert IDs from bytes to str if needed
    ids = test_data[b'ids']
    ids = [id.decode('utf-8') if isinstance(id, bytes) else id for id in ids]

    # Create submission file
    submission = pd.DataFrame({
        'ID': ids,
        'Labels': predictions
    })
    submission.to_csv('submissions3.csv', index=False)
    print("Predictions saved to submissions3.csv")

if __name__ == '__main__':
    make_predictions()