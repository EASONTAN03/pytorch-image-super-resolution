import math
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from ..metrics import PSNR, SSIM
from ..utils import create_base_path
from tqdm import tqdm # Import tqdm
import os
class SRCNN(nn.Module):
    def __init__(self, f1=9, f2=1, f3=5, n1=64, n2=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, n1, f1, padding=(f1 - 1) // 2)
        self.conv2 = nn.Conv2d(n1, n2, f2, padding=(f2 - 1) // 2)
        self.conv3 = nn.Conv2d(n2, 1, f3, padding=(f3 - 1) // 2)
        self.relu = nn.ReLU(inplace=True)
        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # std = math.sqrt(2 / (m.out_channels * m.weight[0][0].numel()))
                std=0.001
                nn.init.normal_(m.weight, 0.0, std)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.conv3.weight, 0.0, 0.001)
        nn.init.zeros_(self.conv3.bias)

def train(train_loader: DataLoader, val_loader: DataLoader, SRCNN_config: dict, save_dirs: dict, SEED: int = 42):
    """
    Trains the SRCNN model and returns the trained model and training logs.

    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        SRCNN_config (dict): Configuration dictionary for SRCNN.
        test_loader (torch.utils.data.DataLoader, optional): DataLoader for test data. Defaults to None.
        SEED (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: A tuple containing:
            - model (nn.Module): The trained SRCNN model.
            - training_logs (dict): A dictionary containing:
                - 'train_losses' (list): List of training loss per epoch.
                - 'val_losses' (list): List of validation loss per epoch.
                - 'val_psnr_scores' (list): List of validation PSNR scores per epoch.
                - 'val_ssim_scores' (list): List of validation SSIM scores per epoch.
                - 'parameters' (dict): The SRCNN model parameters used for training.
    """
    
    if SRCNN_config is None:
        raise ValueError("SRCNN_config must be provided for SRCNN training.")
    
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Extract SRCNN specific parameters from config
    params = SRCNN_config.get('parameters', {})
    f1 = params.get('f1', 9)
    f2 = params.get('f2', 1)
    f3 = params.get('f3', 5)
    n1 = params.get('n1', 64)
    n2 = params.get('n2', 32)

    model_momentum = float(SRCNN_config.get('model_momentum', 0.9))

    epochs = SRCNN_config.get('epochs', 20)

    # Initialize SRCNN model (assuming SRCNN class is accessible)
    model = SRCNN(f1=f1, f2=f2, f3=f3, n1=n1, n2=n2).to(device)
    
    # Loss function (Mean Squared Error for SRCNN)
    criterion = nn.MSELoss().to(device)
    
    # Optimizer (SGD as per common SRCNN implementations, with momentum and weight decay)
    optimizer = torch.optim.SGD(
        [
        {'params': model.conv1.parameters(), 'lr': 1e-4},
        {'params': model.conv2.parameters(), 'lr': 1e-4},
        {'params': model.conv3.parameters(), 'lr': 1e-5},  # lower LR for last layer
        ], 
        lr=SRCNN_config.get('model_lr', 1e-4),
        momentum=model_momentum
    )
    
    psnr_metric = PSNR().to(device)
    ssim_metric = SSIM().to(device)

    print("Starting SRCNN training...")
    best_val_loss = float('inf')

    # Lists to store metrics for returning
    train_losses = []
    val_losses = []
    val_psnr_scores = []
    val_ssim_scores = []

    if save_dirs["resume"] and os.path.exists(save_dirs["resume"]):
        checkpoint_path = save_dirs["resume"] / "SRCNN_checkpoint.pth"
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1 # Start from the next epoch
        best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
        print(f"Resumed from epoch {start_epoch}, with best validation loss: {best_val_loss:.4f}")
    else:
        start_epoch = 0
        checkpoint_path = save_dirs["models"] / "SRCNN_checkpoint.pth"
        print(f"Warning: Resume path '{save_dirs['resume']}' not found. Starting training from scratch.")

    for epoch in range(start_epoch, epochs):
        model.train() # Set model to training mode
        running_loss = 0.0
        start_time = time.time()

        for batch_idx, (lr_images, hr_images) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training", leave=False)):
            # Skip batch if image reading failed in dataset (returned None)
            if lr_images is None or hr_images is None:
                print(f"Skipping batch {batch_idx} due to None images.")
                continue

            lr_images = lr_images.to(device, non_blocking=True)
            hr_images = hr_images.to(device, non_blocking=True)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(lr_images)
            loss = criterion(outputs, hr_images)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * lr_images.size(0)
            print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.6f}")

            # --- Validation after certain batch ---
            if batch_idx % 1000 == 0:
                model.eval()
                val_loss_batch = 0.0
                val_psnr_batch = 0.0
                val_ssim_batch = 0.0
                with torch.no_grad():
                    for lr_images_val, hr_images_val in val_loader:
                        if lr_images_val is None or hr_images_val is None:
                            continue

                        lr_images_val = lr_images_val.to(device, non_blocking=True)
                        hr_images_val = hr_images_val.to(device, non_blocking=True)

                        outputs_val = model(lr_images_val)
                        batch_val_loss = criterion(outputs_val, hr_images_val)
                        val_loss_batch += batch_val_loss.item() * lr_images_val.size(0)

                        val_psnr_batch += psnr_metric(outputs_val, hr_images_val).item() * lr_images_val.size(0)
                        val_ssim_batch += ssim_metric(outputs_val, hr_images_val).item() * lr_images_val.size(0)

                # Average over the entire validation set
                avg_val_loss = val_loss_batch / len(val_loader.dataset)
                avg_val_psnr = val_psnr_batch / len(val_loader.dataset)
                avg_val_ssim = val_ssim_batch / len(val_loader.dataset)

                print(f"→ Validation after batch {batch_idx+1}: "
                    f"Loss: {avg_val_loss:.6f}, PSNR: {avg_val_psnr:.4f}, SSIM: {avg_val_ssim:.4f}")
                model.train()

        end_time = time.time()
        print(f"Epoch [{epoch+1}/{epochs}], Epoch Loss: {epoch_loss:.6f}, Time: {end_time - start_time:.2f}s")

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        end_time = time.time()
        epoch_duration = end_time - start_time

        # Validation phase
        model.eval() # Set model to evaluation mode
        val_running_loss = 0.0
        val_psnr_sum = 0.0
        val_ssim_sum = 0.0
        with torch.no_grad(): # Disable gradient calculation for validation
            for lr_images_val, hr_images_val in val_loader:
                # Skip batch if image reading failed in dataset (returned None)
                if lr_images_val is None or hr_images_val is None:
                    print(f"Skipping validation batch due to None images.")
                    continue

                lr_images_val = lr_images_val.to(device, non_blocking=True)
                hr_images_val = hr_images_val.to(device, non_blocking=True)

                outputs_val = model(lr_images_val)
                val_loss = criterion(outputs_val, hr_images_val)
                val_running_loss += val_loss.item() * lr_images_val.size(0)

                # Calculate PSNR and SSIM for validation
                val_psnr_sum += psnr_metric(outputs_val, hr_images_val).item() * lr_images_val.size(0)
                val_ssim_sum += ssim_metric(outputs_val, hr_images_val).item() * lr_images_val.size(0)
        
        avg_val_loss = val_running_loss / len(val_loader.dataset)
        avg_val_psnr = val_psnr_sum / len(val_loader.dataset)
        avg_val_ssim = val_ssim_sum / len(val_loader.dataset)
        
        val_losses.append(avg_val_loss)
        val_psnr_scores.append(avg_val_psnr)
        val_ssim_scores.append(avg_val_ssim)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_psnr': avg_val_psnr,
                'best_val_ssim': avg_val_ssim
            }
            torch.save(checkpoint, checkpoint_path)

        print(f"Epoch [{epoch+1}/{epochs}], "
            f"Train Loss: {epoch_loss:.6f}, "
            f"Validation Loss: {avg_val_loss:.6f}, "
            f"Validation PSNR: {avg_val_psnr:.4f}, "
            f"Validation SSIM: {avg_val_ssim:.4f}, "
            f"Time: {epoch_duration:.2f}s")
            
    print("SRCNN training finished!")
    
    training_logs = {
        "start_epoch": start_epoch,
        "epochs_run": epochs - start_epoch,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_psnr_scores': val_psnr_scores,
        'val_ssim_scores': val_ssim_scores,
    }

    # Return model and the single training_logs dictionary for clarity
    return model, training_logs