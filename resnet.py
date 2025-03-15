import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
from torchsummary import summary
from tqdm import tqdm
import pickle
import pandas as pd
from PIL import Image

# -----------------------------
# Google Drive Setup for Google Colab
# -----------------------------
# If running in Google Colab, mount the drive and set file paths accordingly.
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_MOUNTED = True
except ImportError:
    DRIVE_MOUNTED = False

if DRIVE_MOUNTED:
    DATA_ROOT = '/content/drive/MyDrive/cifar_data'
    CHECKPOINT_PATH = '/content/drive/MyDrive/cifar_model/checkpoint.pth'
    SNAPSHOT_FOLDER = '/content/drive/MyDrive/cifar_model/snapshots'
else:
    DATA_ROOT = './data'
    CHECKPOINT_PATH = 'checkpoint.pth'
    SNAPSHOT_FOLDER = 'snapshots'

# Set random seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#######################################
# Mixup and CutMix Functions
#######################################
def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation to a batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute the mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix augmentation to a batch."""
    batch_size, _, H, W = x.size()
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(batch_size).to(x.device)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, bby1:bby2, bbx1:bbx2] = x[rand_index, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[rand_index]
    return x, y_a, y_b, lam

#######################################
# Model: Residual Block and Custom ResNet
#######################################
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class CustomResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomResNet, self).__init__()
        # Updated initial convolution: 3 -> 84 channels
        self.conv1 = nn.Conv2d(3, 84, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(84)
        # Residual layers
        self.layer1 = self.make_layer(84, 84, num_blocks=2, stride=1)
        self.layer2 = self.make_layer(84, 168, num_blocks=2, stride=2)
        self.layer3 = self.make_layer(168, 336, num_blocks=2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(336, num_classes)
    
    def make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

#######################################
# Data Loading Function for CIFAR-10 (50k training images)
#######################################
def load_cifar10_data():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), 
                                 ratio=(0.3, 3.3), value='random')
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform_train
    )
    
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)
    return train_loader, val_loader

#######################################
# Training Function with Checkpointing, Early Stopping, and Cosine Annealing
#######################################
def train_model(model, train_loader, val_loader, num_epochs=80, patience=10, checkpoint_path=CHECKPOINT_PATH, snapshot_interval=10):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    T_max = len(train_loader) * num_epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    
    # Check for existing checkpoint to resume training
    start_epoch = 0
    best_val_acc = 0
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint['best_val_acc']
            print(f"Resuming from epoch {start_epoch} with best validation accuracy: {best_val_acc:.2f}%")
        else:
            model.load_state_dict(checkpoint)
            print("Checkpoint loaded (state_dict only). Starting from epoch 0.")
    
    # Create snapshots folder in the designated directory
    snapshot_folder = SNAPSHOT_FOLDER
    os.makedirs(snapshot_folder, exist_ok=True)
    
    epochs_without_improvement = 0
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            # Randomly choose between Mixup and CutMix (50% chance each)
            if np.random.rand() < 0.5:
                images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=0.2)
                optimizer.zero_grad()
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                images, targets_a, targets_b, lam = cutmix_data(images, labels, alpha=1.0)
                optimizer.zero_grad()
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'Loss': train_loss/total, 'Acc': 100.*correct/total})
        
        # Validation phase (no augmentation)
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        val_acc = 100.0 * val_correct / val_total
        print(f'\nEpoch {epoch+1}: Validation Accuracy: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc
            }
            torch.save(checkpoint, checkpoint_path)
            print("Validation accuracy improved, checkpoint saved.")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")
        
        # Save snapshot every snapshot_interval epochs
        if (epoch + 1) % snapshot_interval == 0:
            snapshot_path = os.path.join(snapshot_folder, f'snapshot_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), snapshot_path)
            print(f"Snapshot saved at epoch {epoch+1}")
        
        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

#######################################
# Inference Functions for TTA Ensemble
#######################################
def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict

def prepare_test_data(data):
    """
    data: A NumPy array of shape (N, 32, 32, 3).
    Returns: A torch.FloatTensor of shape (N, 3, 32, 32), normalized.
    """
    data = data.astype(np.float32)
    data = np.transpose(data, (0, 3, 1, 2))  # (N, 3, 32, 32)
    data /= 255.0
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32).reshape(1, 3, 1, 1)
    data = (data - mean) / std
    return torch.from_numpy(data)

def visualize_samples(images, num_samples=5):
    """
    Visualize a few sample images from the preprocessed tensor.
    """
    images = images.clone().numpy()
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
    std = np.array([0.2023, 0.1994, 0.2010]).reshape(3, 1, 1)
    images = images * std + mean
    images = np.clip(images, 0, 1)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(num_samples*2, 2))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(np.transpose(images[i], (1, 2, 0)))
        plt.axis('off')
    plt.suptitle("Sample Preprocessed Custom Test Images")
    plt.show()

def tta_predict(model, img, base_transform, tta_transforms, device, num_augments=5):
    """
    Apply TTA on a single PIL image and return averaged prediction.
    """
    model.eval()
    predictions = []
    # Base prediction (without extra augmentation)
    img_tensor = base_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
    predictions.append(output.cpu().numpy())
    
    # Apply TTA transforms
    for t in tta_transforms:
        for _ in range(num_augments):
            aug_transform = transforms.Compose([
                t,
                base_transform
            ])
            aug_tensor = aug_transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(aug_tensor)
            predictions.append(output.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    avg_pred = np.mean(predictions, axis=0)
    return avg_pred

def ensemble_inference(model_paths, test_data_tensor, device, batch_size=100, tta=False, base_transform=None, tta_transforms=None, num_augments=5):
    """
    Performs inference by averaging predictions from multiple snapshots.
    Optionally applies TTA if tta=True.
    model_paths: list of paths to snapshot checkpoints.
    test_data_tensor: tensor of test images of shape (N, 3, 32, 32).
    """
    all_preds = []
    for path in model_paths:
        model = CustomResNet(num_classes=10).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        preds = []
        if tta:
            # For TTA, we assume test_data_tensor is from prepare_test_data (so it's a tensor).
            # We need to convert each sample back to PIL.
            for i in range(0, test_data_tensor.size(0), batch_size):
                batch = test_data_tensor[i:i+batch_size]
                # We'll convert each image in the batch to PIL, apply TTA, and get prediction.
                for img_tensor in batch:
                    img = transforms.ToPILImage()(img_tensor.cpu())
                    pred = tta_predict(model, img, base_transform, tta_transforms, device, num_augments)
                    preds.append(pred)
            preds = np.array(preds)  # shape (N, num_classes)
        else:
            with torch.no_grad():
                loader = DataLoader(test_data_tensor, batch_size=batch_size, shuffle=False)
                for batch in loader:
                    batch = batch.to(device)
                    output = model(batch)
                    preds.append(output.cpu().numpy())
            preds = np.concatenate(preds, axis=0)
        all_preds.append(preds)
    # Average predictions across models
    ensemble_preds = np.mean(np.stack(all_preds, axis=0), axis=0)
    final_labels = np.argmax(ensemble_preds, axis=1)
    return final_labels

#######################################
# Inference Function for Competition Test Set
#######################################
def inference_competition():
    # Load competition test data from pickle file
    with open('cifar_test_nolabel.pkl', 'rb') as fo:
        test_data_dict = pickle.load(fo, encoding='bytes')
    test_data_raw = test_data_dict[b'data']  # assumed shape (N, 32, 32, 3)
    test_ids = test_data_dict[b'ids']
    test_images = prepare_test_data(test_data_raw)
    
    # Visualize a few samples
    visualize_samples(test_images, num_samples=5)
    
    # Define base transform and TTA transforms for inference (PIL format)
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010))
    ])
    tta_transforms = [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(10)
    ]
    
    # Get list of snapshot checkpoints from the snapshots folder
    model_paths = []
    if os.path.exists(SNAPSHOT_FOLDER):
        for f in sorted(os.listdir(SNAPSHOT_FOLDER)):
            if f.endswith('.pth'):
                model_paths.append(os.path.join(SNAPSHOT_FOLDER, f))
    # Also include the best checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        model_paths.append(CHECKPOINT_PATH)
    if len(model_paths) == 0:
        print("No snapshots found. Exiting inference.")
        return
    
    # Perform ensemble inference with TTA
    final_preds = ensemble_inference(model_paths, test_images, device, batch_size=100, tta=True, base_transform=base_transform, tta_transforms=tta_transforms, num_augments=5)
    
    # Convert IDs from bytes to strings if necessary
    ids = [id.decode('utf-8') if isinstance(id, bytes) else id for id in test_ids]
    
    submission = pd.DataFrame({
        'ID': ids,
        'Labels': final_preds
    })
    submission.to_csv('submissions_ensemble.csv', index=False)
    print("Submission saved to submissions_ensemble.csv")

#######################################
# Main Execution
#######################################
def main():
    # Set these variables as desired
    mode = 'train'      # Change to 'inference' for ensemble inference
    epochs = 150         # Total number of training epochs
    patience = 25.       # Patience for early stopping
    
    if mode == 'train':
        model = CustomResNet(num_classes=10).to(device)
        print("Model architecture:")
        summary(model, (3, 32, 32))
        train_loader, val_loader = load_cifar10_data()
        train_model(model, train_loader, val_loader, num_epochs=epochs, patience=patience, checkpoint_path=CHECKPOINT_PATH, snapshot_interval=10)
    else:
        inference_competition()

# Run the main function
main()