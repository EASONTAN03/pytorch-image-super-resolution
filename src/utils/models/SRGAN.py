import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
import os
import time
import csv
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from ..metrics import PSNR, SSIM
from ..utils import *
import tqdm

class ResidualBlock(nn.Module):
    """
    A single Residual Block as described in the SRGAN paper.
    Consists of two convolutional layers with BatchNorm and PReLU,
    and a skip connection.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels, momentum=0.5) # Keras default momentum is 0.9, PyTorch default is 0.1
        self.prelu1 = nn.PReLU() # PyTorch's PReLU implicitly shares across spatial dims if num_parameters=1 (default)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, momentum=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x # Store input for skip connection

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = torch.add(residual, out) # Or simply residual + out
        return out
        
class UpsampleBlock(nn.Module):
    """
    Upsampling block. Uses Conv2d followed by nn.Upsample and PReLU.
    """
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()
        # This convolution should output `out_channels`
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.prelu = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.upsample(out)
        out = self.prelu(out)
        return out


class DiscriminatorBlock(nn.Module):
    """
    A single block for the Discriminator.
    Consists of Conv2d, optional BatchNorm, and LeakyReLU.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_bn: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.8) if use_bn else None
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.bn is not None:
            out = self.bn(out)
        out = self.leaky_relu(out)
        return out
    
class VGGFeatureExtractor(nn.Module):
    """
    VGG19 based feature extractor for perceptual loss.
    Extracts features from an intermediate layer (e.g., conv5_4).
    """
    def __init__(self, feature_layer: str = 'features.35'): # Corresponds to VGG19 conv5_4 layer
        super().__init__()
        vgg_model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        # We only need the 'features' part of VGG
        # Define a list of layer names/indices to keep
        # For conv5_4, you might need to go up to index 35 (inclusive) in VGG19.features
        # Depending on the exact layer you want, adjust this.
        # Layer 35 is just before the last maxpool, often used for content loss.
        self.features = nn.Sequential(*list(vgg_model.features)[:36]) # Keep layers up to and including index 35 (conv5_4)
        
        # Freeze VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize input to VGG (standard practice for pre-trained models)
        # VGG expects images normalized to [0, 1] then mean/std normalized
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std

        return self.features(x)


# --- Main Networks ---
class Generator(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, num_res_blocks: int = 16, upscale_factor: int = 4):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=9, padding=4, bias=False)
        self.prelu1 = nn.PReLU()

        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks)])

        # After residual blocks (from create_gen's 'layers = Conv2D...' block after loop)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.5)

        # Upsampling blocks (for X4, two blocks are needed, each performing 2x upsampling)
        upsample_layers = []
        # Calculate how many 2x upsampling steps are needed (e.g., log2(4) = 2 steps)
        num_upsample_steps = int(torch.log2(torch.tensor(upscale_factor)).item())

        for _ in range(num_upsample_steps):
            # Each UpsampleBlock takes 64 channels and outputs 64 channels
            # It performs 2x spatial upsampling
            upsample_layers.append(UpsampleBlock(64, 64, scale_factor=2)) # <--- CRITICAL CHANGE HERE
        self.upsample_blocks = nn.Sequential(*upsample_layers)

        # Final convolution
        # This now correctly expects 64 channels from the last upsample block
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=9, padding=4) # <--- CRITICAL CHANGE HERE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial part
        out = self.conv1(x)
        out = self.prelu1(out)
        
        # Store output for global skip connection
        initial_features = out

        # Residual blocks
        out = self.res_blocks(out)

        # After residual blocks, add initial_features (global skip connection)
        out = self.conv2(out)
        out = self.bn2(out)
        out = torch.add(out, initial_features) # add([layers, temp])

        # Upsampling
        out = self.upsample_blocks(out)

        # Final convolution
        out = self.final_conv(out)
        # SRGAN paper typically uses tanh activation here for output between -1 and 1
        # For images in [0,1], a sigmoid or no activation (and then clamp/scale) is used.
        # Based on your Keras code not having activation, will omit for now.
        return out

class Discriminator(nn.Module):
    """
    The Discriminator network for SRGAN.
    Takes a high-resolution image and outputs a probability of it being real/fake.
    """
    # Added hr_image_size to make the discriminator flexible
    def __init__(self, in_channels: int = 3, hr_image_size: int = 96):
        super().__init__()
        df = 64 # Base filter size

        self.block1 = DiscriminatorBlock(in_channels, df, stride=1, use_bn=False)
        self.block2 = DiscriminatorBlock(df, df, stride=2)
        self.block3 = DiscriminatorBlock(df, df * 2, stride=1)
        self.block4 = DiscriminatorBlock(df * 2, df * 2, stride=2)
        self.block5 = DiscriminatorBlock(df * 2, df * 4, stride=1)
        self.block6 = DiscriminatorBlock(df * 4, df * 4, stride=2)
        self.block7 = DiscriminatorBlock(df * 4, df * 8, stride=1)
        self.block8 = DiscriminatorBlock(df * 8, df * 8, stride=2)

        # Calculate the final spatial dimension after all downsampling blocks
        # There are 4 DiscriminatorBlocks with stride=2 (block2, block4, block6, block8).
        # This means the spatial dimensions (height and width) are divided by 2^4 = 16.
        final_spatial_dim = hr_image_size // (2**4) # Use integer division

        # Calculate the number of input features for the first dense layer.
        # This is (channels_out_of_last_conv_block) * (final_height) * (final_width).
        final_channels = df * 8 # Output channels of self.block8

        self.in_features = final_channels * final_spatial_dim * final_spatial_dim

        self.flatten = nn.Flatten()
        # The input features for the first dense layer are now dynamically calculated
        self.dense1 = nn.Linear(self.in_features, df * 16)
        self.leaky_relu_dense = nn.LeakyReLU(negative_slope=0.2)
        self.output_dense = nn.Linear(df * 16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)
        out = self.block8(out)

        out = self.flatten(out)
        out = self.dense1(out)
        out = self.leaky_relu_dense(out)
        out = self.output_dense(out)
        validity = self.sigmoid(out)
        return validity

class SRGAN(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, 
                 num_res_blocks: int = 16, upscale_factor: int = 4,
                 hr_shape: tuple = (3, 96, 96)): # hr_shape for VGG input (C, H, W)
        super().__init__()

        self.generator = Generator(in_channels, out_channels, num_res_blocks, upscale_factor)
        self.discriminator = Discriminator(in_channels=out_channels) # Discriminator takes HR image (either real or generated)
        self.vgg_feature_extractor = VGGFeatureExtractor() # VGG assumes 3-channel input

    def forward(self, lr_images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the generator part, generating HR images from LR.
        """
        return self.generator(lr_images)

    # Note: In a full training setup, you would have separate forward passes for
    # training the discriminator and generator (with perceptual + adversarial loss).
    # The `build_vgg` method you had would be integrated into the VGGFeatureExtractor.
    # The `create_gen` and `create_dics` functions have been turned into Generator and Discriminator classes.

def train(train_loader: DataLoader, valid_loader: DataLoader, SRGAN_config: dict, save_dir: dict, SEED: int):
    """
    Trains the SRGAN model.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        valid_loader (DataLoader): DataLoader for validation data.
        SRGAN_config (dict): Training configuration dictionary.
        save_dir (dict): Dictionary containing paths for saving models and logs.
                          Expected keys: 'models', 'resume', 'logs'.
        SEED (int): Random seed for reproducibility.
    """
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the SRGAN model with parameters from config
    model = SRGAN(
        in_channels=SRGAN_config.get('in_channels', 3),
        out_channels=SRGAN_config.get('out_channels', 3),
        num_res_blocks=SRGAN_config.get('num_res_blocks', 16),
        upscale_factor=SRGAN_config.get('upscale_factor', 4),
        hr_shape=tuple(SRGAN_config.get('hr_shape', (3, 96, 96)))
    ).to(device)

    generator = model.generator.to(device)
    discriminator = model.discriminator.to(device)
    vgg_feature_extractor = model.vgg_feature_extractor.to(device)
    vgg_feature_extractor.eval() # VGG is used for feature extraction, keep in eval mode

    # Extract hyperparameters from config
    epochs = SRGAN_config['epochs']
    param = SRGAN_config.get('parameters', {})
    lr_g = float(param.get('learning_rate_generator', 1e-4))
    lr_d = float(param.get('learning_rate_discriminator', 1e-4))
    b1 = float(param.get('b1', 0.9))
    b2 = float(param.get('b2', 0.999))
    lambda_adv = float(param.get('lambda_adversarial', 1e-3))
    lambda_content = float(param.get('lambda_content', 1.0))

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr_g, betas=(b1, b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(b1, b2))

    # Loss functions
    criterion_adversarial = nn.BCEWithLogitsLoss()
    criterion_content = nn.MSELoss()

    # Metrics for validation
    psnr_metric = PSNR().to(device)
    ssim_metric = SSIM().to(device)

    # Training logs storage
    train_losses_g = []
    train_losses_d = []
    val_losses_g = []
    val_losses_d = []
    val_psnr_scores = []
    val_ssim_scores = []
    epoch_durations = []

    best_val_psnr = -1.0 # Initialize with a low value for comparison
    best_val_loss_at_best_psnr = float('inf') # Keep track of G loss at best PSNR

    # Checkpoint path for saving the best model
    checkpoint_path = save_dir["models"] / "SRGAN_checkpoint.pth"

    # Resume training if checkpoint exists
    start_epoch = 0
    if save_dir['resume']!=None and os.path.exists(save_dir["resume"]):
        checkpoint_path = save_dir["resume"] / "SRGAN_checkpoint.pth"
        if os.path.exists(checkpoint_path):
            print(f"Resuming training from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            generator.load_state_dict(checkpoint['generator_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_psnr = checkpoint.get('best_val_psnr', best_val_psnr)
            best_val_loss_at_best_psnr = checkpoint.get('best_val_content_loss_at_best_psnr', best_val_loss_at_best_psnr)
            print(f"Resumed from epoch {start_epoch}, with best validation PSNR: {best_val_psnr:.4f}")
        else:
            print(f"Warning: Resume path specified, but checkpoint '{checkpoint_path}' not found. Starting training from scratch.")
    else:
        print(f"Warning: Resume path not specified or not found. Starting training from scratch.")

    print(f"Training SRGAN for {epochs} epochs on {device}")
    print(f"Generator LR: {lr_g}, Discriminator LR: {lr_d}")
    print(f"Adversarial Loss Weight (lambda_adv): {lambda_adv}, Content Loss Weight (lambda_content): {lambda_content}")

    total_start_time = time.time()

    for epoch in range(start_epoch, epochs):
        generator.train()
        discriminator.train()
        epoch_start_time = time.time()
        running_loss_g = 0.0
        running_loss_d = 0.0

        train_loop = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training", leave=False)
        for batch_idx, (lr_images, hr_images) in enumerate(train_loop):
            if lr_images is None or hr_images is None:
                train_loop.write(f"Skipping training batch {batch_idx+1} due to None images.")
                continue

            lr_images = lr_images.to(device, non_blocking=True)
            hr_images = hr_images.to(device, non_blocking=True)

            if lr_images.max() > 1.0 + 1e-5:
                lr_images = lr_images / 255.0
                hr_images = hr_images / 255.0

            # ---------------------
            #   Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            fake_hr_images = generator(lr_images).detach() # Detach to stop gradient flow to Generator

            real_pred = discriminator(hr_images)
            fake_pred = discriminator(fake_hr_images)

            real_labels = torch.ones_like(real_pred).to(device) * 0.9
            fake_labels = torch.zeros_like(fake_pred).to(device) * 0.1

            loss_real = criterion_adversarial(real_pred, real_labels)
            loss_fake = criterion_adversarial(fake_pred, fake_labels)
            d_loss = (loss_real + loss_fake) / 2

            d_loss.backward()
            optimizer_D.step()
            running_loss_d += d_loss.item()

            # -----------------
            #   Train Generator
            # -----------------
            optimizer_G.zero_grad()
            gen_hr_images = generator(lr_images)

            gen_fake_pred = discriminator(gen_hr_images)
            g_adversarial_loss = criterion_adversarial(gen_fake_pred, torch.ones_like(gen_fake_pred).to(device))

            # Prepare images for VGG (ensure 3 channels)
            if gen_hr_images.shape[1] == 1:
                gen_hr_images_vgg = gen_hr_images.repeat(1, 3, 1, 1)
                hr_images_vgg = hr_images.repeat(1, 3, 1, 1)
            else:
                gen_hr_images_vgg = gen_hr_images
                hr_images_vgg = hr_images

            gen_features = vgg_feature_extractor(gen_hr_images_vgg)
            hr_features = vgg_feature_extractor(hr_images_vgg).detach() # Detach HR features as fixed targets

            g_content_loss = criterion_content(gen_features, hr_features)

            g_loss = lambda_adv * g_adversarial_loss + lambda_content * g_content_loss

            g_loss.backward()
            optimizer_G.step()
            running_loss_g += g_loss.item()

            train_loop.set_postfix(G_Loss=g_loss.item(), D_Loss=d_loss.item())


        epoch_duration = time.time() - epoch_start_time
        epoch_durations.append(epoch_duration)
        avg_train_loss_g = running_loss_g / len(train_loader)
        avg_train_loss_d = running_loss_d / len(train_loader)
        train_losses_g.append(avg_train_loss_g)
        train_losses_d.append(avg_train_loss_d)
        
        # ---------------------
        #   Validation Step
        # ---------------------
        generator.eval()
        discriminator.eval()

        val_psnr_sum = 0.0
        val_ssim_sum = 0.0
        val_samples_count = 0
        val_running_g_loss = 0.0
        val_running_d_loss = 0.0

        with torch.no_grad():
            for val_batch_idx, (lr_images_val, hr_images_val) in enumerate(tqdm.tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} - Validating", leave=False)):
                if lr_images_val is None or hr_images_val is None:
                    tqdm.write(f"Skipping validation batch {val_batch_idx+1} due to None images.")
                    continue

                lr_images_val = lr_images_val.to(device, non_blocking=True)
                hr_images_val = hr_images_val.to(device, non_blocking=True)

                if lr_images_val.max() > 1.0 + 1e-5:
                    lr_images_val = lr_images_val / 255.0
                    hr_images_val = hr_images_val / 255.0

                gen_hr_images_val = generator(lr_images_val)
                gen_hr_images_val = torch.clamp(gen_hr_images_val, 0.0, 1.0) # Clamp for accurate PSNR/SSIM
                gen_hr_images_val_y = rgb2yCbCr(gen_hr_images_val)
                hr_images_val_y = rgb2yCbCr(hr_images_val)

                val_psnr_sum += psnr_metric(gen_hr_images_val_y, hr_images_val_y).item() * lr_images_val.size(0)
                val_ssim_sum += ssim_metric(gen_hr_images_val_y, hr_images_val_y).item() * lr_images_val.size(0)
                val_samples_count += lr_images_val.size(0)

                # Calculate Generator's validation loss
                if gen_hr_images_val.shape[1] == 1:
                    gen_hr_images_val_vgg = gen_hr_images_val.repeat(1, 3, 1, 1)
                    hr_images_val_vgg = hr_images_val.repeat(1, 3, 1, 1)
                else:
                    gen_hr_images_val_vgg = gen_hr_images_val
                    hr_images_val_vgg = hr_images_val

                # val_gen_fake_pred = discriminator(gen_hr_images_val)
                # val_g_adversarial_loss = criterion_adversarial(val_gen_fake_pred, torch.ones_like(val_gen_fake_pred).to(device))
                val_gen_features = vgg_feature_extractor(gen_hr_images_val_vgg)
                val_hr_features = vgg_feature_extractor(hr_images_val_vgg)
                val_g_content_loss = criterion_content(val_gen_features, val_hr_features)
                # val_g_loss = lambda_adv * val_g_adversarial_loss + lambda_content * val_g_content_loss
                # val_running_g_loss += val_g_loss.item() * lr_images_val.size(0)
                val_running_g_loss+=val_g_content_loss.item()

                # Calculate Discriminator's validation loss
                # val_real_pred = discriminator(hr_images_val)
                # val_fake_pred = discriminator(gen_hr_images_val.detach())
                # val_loss_real_d = criterion_adversarial(val_real_pred, torch.ones_like(val_real_pred).to(device) * 0.9)
                # val_loss_fake_d = criterion_adversarial(val_fake_pred, torch.zeros_like(val_fake_pred).to(device) * 0.1)
                # val_d_loss = (val_loss_real_d + val_loss_fake_d) / 2
                # val_running_d_loss += val_d_loss.item() * lr_images_val.size(0)



        avg_val_psnr = val_psnr_sum / val_samples_count if val_samples_count > 0 else 0.0
        avg_val_ssim = val_ssim_sum / val_samples_count if val_samples_count > 0 else 0.0
        avg_val_g_loss = val_running_g_loss / val_samples_count if val_samples_count > 0 else 0.0
        # avg_val_d_loss = val_running_d_loss / val_samples_count if val_samples_count > 0 else 0.0

        val_psnr_scores.append(avg_val_psnr)
        val_ssim_scores.append(avg_val_ssim)
        val_losses_g.append(avg_val_g_loss)  # Store G loss for validation
        # val_losses_d.append(avg_val_d_loss)  # Store D loss for validation

        # Save the best Generator model based on validation PSNR
        if avg_val_psnr > best_val_psnr:
            best_val_psnr = avg_val_psnr
            best_val_loss_at_best_psnr = avg_val_g_loss # Store the G loss at this best PSNR
            checkpoint = {
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(), # Save D's state too
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'best_val_psnr': best_val_psnr,
                'best_val_ssim': avg_val_ssim,
                'best_val_content_loss_at_best_psnr': best_val_loss_at_best_psnr
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"--> Saved best model checkpoint to {checkpoint_path} (PSNR: {best_val_psnr:.4f})")

        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train G Loss: {avg_train_loss_g:.6f}, Train D Loss: {avg_train_loss_d:.6f}, "
              f"Validation G Loss: {avg_val_g_loss:.6f}"
              f"Validation PSNR: {avg_val_psnr:.4f}, Validation SSIM: {avg_val_ssim:.4f}, "
              f"Time: {epoch_duration:.2f}s")

    total_end_time = time.time()
    print(f"SRGAN training finished! Total time: {total_end_time - total_start_time:.2f}s")

    training_logs = {
        "start_epoch": start_epoch,
        "epochs_run": epochs - start_epoch,
        'train_losses_generator': train_losses_g,
        'train_losses_discriminator': train_losses_d,
        "val_losses_generator": val_losses_g,
        'val_psnr_scores': val_psnr_scores,
        'val_ssim_scores': val_ssim_scores,
    }
    
    return model, training_logs