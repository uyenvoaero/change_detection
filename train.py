import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from datetime import datetime

# Custom Dataset for LEVIR-CD
class LEVIRCDDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.image_pairs = []
        self.labels = []
        
        # Load image pairs and labels
        img_a_paths = sorted(glob.glob(os.path.join(self.root_dir, 'A', '*.png')))
        img_b_paths = sorted(glob.glob(os.path.join(self.root_dir, 'B', '*.png')))
        label_paths = sorted(glob.glob(os.path.join(self.root_dir, 'label', '*.png')))
        
        for a, b, lbl in zip(img_a_paths, img_b_paths, label_paths):
            self.image_pairs.append((a, b))
            # Label: 0 (unchanged, similar), 1 (changed, different)
            label_img = Image.open(lbl).convert('L')
            label = 1 if np.any(np.array(label_img) > 0) else 0
            self.labels.append(label)
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        img_a_path, img_b_path = self.image_pairs[idx]
        label = self.labels[idx]
        
        img_a = Image.open(img_a_path).convert('RGB')
        img_b = Image.open(img_b_path).convert('RGB')
        
        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
        
        return img_a, img_b, torch.tensor(label, dtype=torch.float32)

# Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Use ResNet18 backbone with updated weights parameter
        self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()  # Remove final FC layer
        self.fc = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward_one(self, x):
        return self.backbone(x)
    
    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        combined = torch.cat((out1, out2), dim=1)
        return self.fc(combined)

# Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output, label):
        euclidean_distance = output.squeeze()
        loss_similar = label * torch.pow(euclidean_distance, 2)
        loss_different = (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        loss = 0.5 * (loss_similar + loss_different)
        return loss.mean()

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, exp_dir):
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for img_a, img_b, labels in train_loader:
            img_a, img_b, labels = img_a.to(device), img_b.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(img_a, img_b)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = (outputs.squeeze() < 0.5).float()  # Threshold at 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_loss /= len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for img_a, img_b, labels in val_loader:
                img_a, img_b, labels = img_a.to(device), img_b.to(device), labels.to(device)
                outputs = model(img_a, img_b)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = (outputs.squeeze() < 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save models
        torch.save(model.state_dict(), os.path.join(exp_dir, 'models', 'last_model.pth'))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(exp_dir, 'models', 'best_model.pth'))
    
    # Plot and save loss/accuracy graphs
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'graphs', 'training_metrics.png'))
    plt.close()

def main():
    # Configuration
    dataset_root = './dataset'
    output_root = './output'
    num_epochs = 40
    batch_size = 16
    learning_rate = 0.0001
    device = torch.device('cpu')
    
    # Create experiment directory
    exp_no = 1
    while os.path.exists(os.path.join(output_root, f'exp{exp_no}')):
        exp_no += 1
    exp_dir = os.path.join(output_root, f'exp{exp_no}')
    os.makedirs(os.path.join(exp_dir, 'graphs'))
    os.makedirs(os.path.join(exp_dir, 'models'))
    os.makedirs(os.path.join(exp_dir, 'inference_results'))
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets and dataloaders
    train_dataset = LEVIRCDDataset(dataset_root, 'train', transform)
    val_dataset = LEVIRCDDataset(dataset_root, 'val', transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model, loss, optimizer
    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, exp_dir)

if __name__ == '__main__':
    main()