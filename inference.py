import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import glob
from train import SiameseNetwork, LEVIRCDDataset

def run_inference(model, test_loader, device, output_dir, batch_size):
    model.eval()
    # Create separate directories for correct and wrong predictions
    correct_dir = os.path.join(output_dir, 'correct_predictions')
    wrong_dir = os.path.join(output_dir, 'wrong_predictions')
    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(wrong_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, (img_a, img_b, labels) in enumerate(test_loader):
            img_a, img_b, labels = img_a.to(device), img_b.to(device), labels.to(device)
            outputs = model(img_a, img_b)
            preds = (outputs.squeeze() < 0.5).float()  # Threshold at 0.5
            
            for j in range(img_a.size(0)):
                label = labels[j].item()
                pred = preds[j].item()
                
                # Convert images back to displayable format
                img_a_np = img_a[j].cpu().permute(1, 2, 0).numpy()
                img_b_np = img_b[j].cpu().permute(1, 2, 0).numpy()
                img_a_np = (img_a_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
                img_b_np = (img_b_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
                
                # Create figure with two subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                
                # Display images
                ax1.imshow(img_a_np)
                ax1.set_title("Image A")
                ax1.axis("off")
                
                ax2.imshow(img_b_np)
                ax2.set_title("Image B")
                ax2.axis("off")
                
                # Add text annotation above subplots
                is_correct = pred == label
                truth_text = "Unchange" if label == 0 else "Change"
                predict_text = "Unchange" if pred == 0 else "Change"
                correct_text = "Correct prediction" if is_correct else "Wrong prediction"
                plt.suptitle(f"[{correct_text}] Truth: {truth_text} | Predict: {predict_text}", fontsize=14)
                
                # Save the figure in the appropriate directory based on prediction correctness
                img_a_path = test_loader.dataset.image_pairs[i * batch_size + j][0]
                filename = os.path.basename(img_a_path)
                save_dir = correct_dir if is_correct else wrong_dir
                plt.savefig(os.path.join(save_dir, filename), bbox_inches="tight")
                
                # Close the figure to free memory
                plt.close(fig)

def main():
    # Configuration
    dataset_root = './dataset'
    output_root = './output'
    batch_size = 16
    device = torch.device('cpu')
    
    # Find latest experiment
    exp_dirs = sorted([d for d in glob.glob(os.path.join(output_root, 'exp*'))])
    if not exp_dirs:
        raise ValueError("No experiment directories found.")
    exp_dir = exp_dirs[-1]
    output_dir = os.path.join(exp_dir, 'inference_results')
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test dataset and dataloader
    test_dataset = LEVIRCDDataset(dataset_root, 'test', transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Load model
    model = SiameseNetwork().to(device)
    model_path = os.path.join(exp_dir, 'models', 'best_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Run inference
    run_inference(model, test_loader, device, output_dir, batch_size)

if __name__ == '__main__':
    main()