# PyTorch Image Super-Resolution

PyTorch reimplementation of SRCNN and SRGAN for single-image super-resolution on the DIV2K dataset. Features a modular codebase with shared components for fair comparison between CNN- and GAN-based methods. Includes training, evaluation, and reproducibility support to validate results against the original papers.

## 🚀 Features

- **Multiple Models**: SRCNN (CNN-based) and SRGAN (GAN-based) implementations
- **Modular Design**: Shared components for fair model comparison
- **TensorBoard Integration**: Real-time training monitoring and visualization
- **CUDA Support**: GPU acceleration for faster training and inference
- **Comprehensive Evaluation**: PSNR and SSIM metrics for quality assessment
- **Flexible Dataset**: Support for DIV2K training and Set5 validation
- **Configuration Management**: YAML-based configuration for easy experimentation

## 📁 Project Structure

```
pytorch-image-super-resolution/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── config.yaml                  # Training configuration
├── make_dataset.py             # Dataset preprocessing script
├── .cursorignore               # Cursor IDE ignore file
│
├── src/                        # Source code directory
│   ├── train.py                # Main training script
│   ├── evaluate.py             # Model evaluation script
│   ├── dataset.py              # Dataset loading and preprocessing
│   │
│   ├── models/                 # Model architectures
│   │   ├── SRCNN.py           # SRCNN implementation
│   │   └── SRGAN.py           # SRGAN implementation
│   │
│   └── utils/                  # Utility functions
│       ├── dataset.py          # Dataset utilities
│       ├── imgproc.py          # Image processing functions
│       └── model.py            # Model utilities
│
├── dataset/                    # Dataset directory
│   ├── raw/                    # Raw dataset files
│   │   ├── DIV2K/             # DIV2K dataset
│   │   │   ├── HR/            # High-resolution images
│   │   │   │   ├── train/     # Training images
│   │   │   │   └── valid/     # Validation images
│   │   │   └── LR/            # Low-resolution images (generated)
│   │   └── Set5/              # Set5 test dataset
│   │
│   └── prepared/               # Processed dataset
│       ├── DIV2K/             # Processed DIV2K
│       │   ├── hr/            # High-resolution patches (96x96)
│       │   └── lr/            # Low-resolution patches (24x24)
│       └── Set5/              # Processed Set5
│           └── valid/         # Validation set
│
├── backup/                     # Backup files
│   ├── make_dataset.py        # Backup dataset script
│   └── make_dataset copy.py   # Dataset script copy
│
├── documents/                  # Research papers and documentation
│   ├── 1501.00092v3.pdf       # SRCNN paper
│   ├── 1609.04802v5.pdf       # SRGAN paper
│   ├── 2006.13846v2.pdf       # Additional research
│   ├── Notes and Plan.docx    # Project notes
│   └── ...                    # Other research papers
│
├── samples/                    # Generated during training
│   ├── logs/                  # TensorBoard logs
│   └── {exp_name}/            # Experiment-specific outputs
│
└── results/                    # Training results
    └── {exp_name}/            # Model checkpoints and outputs
        ├── best.pth.tar       # Best model checkpoint
        └── last.pth.tar       # Latest model checkpoint
```

## 🛠️ Installation

### 1. Environment Setup

```bash
# Create conda environment
conda create -n pytorch-image-super-resolution python==3.11
conda activate pytorch-image-super-resolution

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 2. Required Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
tensorboard>=2.13.0
opencv-python>=4.7.0
numpy>=1.24.0
pyyaml>=6.0
pillow>=9.5.0
```

## 📊 Dataset Setup

### Dataset Structure

The project expects the following dataset structure:

```
dataset/
├── raw/
│   ├── DIV2K/
│   │   └── HR/
│   │       ├── train/          # DIV2K training images
│   │       └── valid/          # DIV2K validation images
│   └── Set5/                   # Set5 test images (ground truth)
│
└── prepared/
    ├── DIV2K/
    │   ├── hr/                 # 96x96 HR patches
    │   └── lr/                 # 24x24 LR patches (X4 downscale)
    └── Set5/
        └── valid/              # Processed Set5 validation
```

### Dataset Preprocessing

Generate training patches from raw images:

```bash
# Linux/macOS
python make_dataset.py \
  --train_dir dataset/raw/DIV2K/HR/train \
  --valid_dir dataset/raw/DIV2K/HR/valid \
  --output_dir dataset/prepared/DIV2K \
  --crop_size 96 \
  --stride 48 \
  --upscale_factor 4 \
  --interpolation cubic

# Windows PowerShell
python make_dataset.py `
  --train_dir dataset/raw/DIV2K/HR/train `
  --valid_dir dataset/raw/DIV2K/HR/valid `
  --output_dir dataset/prepared/DIV2K `
  --crop_size 96 `
  --stride 48 `
  --upscale_factor 4 `
  --interpolation cubic
```

## ⚙️ Configuration

The `config.yaml` file contains all training parameters:

```yaml
# Model selection: "SRCNN" or "SRGAN"
model: "SRCNN"

# Dataset configuration
dataset:
  train_image_dir: "dataset/prepared/DIV2K/hr"
  test_lr_image_dir: "dataset/prepared/Set5/valid"
  test_hr_image_dir: "dataset/raw/Set5"
  image_size: 96
  upscale_factor: 4
  batch_size: 16
  num_workers: 4

# Training parameters
train:
  epochs: 1000
  learning_rate: 1e-4
  momentum: 0.9
  weight_decay: 1e-4
  print_frequency: 100

# Device configuration
device: "cuda"  # or "cpu"
```

## 🚀 Usage

### Training

Train SRCNN model:
```bash
python src/train.py
```

The training script will:
- Load configuration from `config.yaml`
- Initialize the selected model (SRCNN or SRGAN)
- Set up data loaders for training and validation
- Start training with TensorBoard logging
- Save model checkpoints in `results/` directory

### Monitoring Training

View training progress with TensorBoard:
```bash
tensorboard --logdir samples/logs
```

### Evaluation

Evaluate trained model:
```bash
python src/evaluate.py --model_path results/best.pth.tar
```

## 🏗️ Model Architectures

### SRCNN (Super-Resolution Convolutional Neural Network)

- **Architecture**: 3-layer CNN
- **Input**: 24x24 LR Y-channel images
- **Output**: 96x96 SR Y-channel images
- **Upscaling**: 4x super-resolution
- **Loss**: Mean Squared Error (MSE)

### SRGAN (Super-Resolution Generative Adversarial Network)

- **Generator**: ResNet-based architecture with 16 residual blocks
- **Discriminator**: VGG-inspired discriminator network
- **Upscaling**: 4x super-resolution using sub-pixel convolution
- **Loss**: Adversarial loss + Content loss (VGG features) + MSE loss

## 📈 Key Components

### Dataset Loading (`src/dataset.py`)
- `TrainValidImageDataset`: Handles training/validation data loading
- `TestImageDataset`: Handles test data loading
- `CUDAPrefetcher`: Accelerates data loading using CUDA streams

### Image Processing (`src/utils/imgproc.py`)
- Color space conversion (BGR ↔ YCbCr)
- Image resizing and cropping
- Tensor conversion utilities

### Training Loop (`src/train.py`)
- Mixed precision training with AMP
- TensorBoard logging
- Model checkpointing
- PSNR/SSIM evaluation

## 🔧 Customization

### Adding New Models

1. Create model file in `src/models/`
2. Implement PyTorch `nn.Module` class
3. Add model import and initialization in `src/train.py`
4. Update `config.yaml` with model-specific parameters

### Modifying Training Parameters

Edit `config.yaml` to adjust:
- Learning rate and optimization parameters
- Batch size and data loading settings
- Model architecture parameters
- Training duration and checkpointing

## 📊 Expected Results

### SRCNN
- **PSNR**: ~30.48 dB on Set5
- **SSIM**: ~0.8626 on Set5
- **Training Time**: ~2-3 hours on RTX 3080

### SRGAN
- **PSNR**: ~29.40 dB on Set5
- **SSIM**: ~0.8472 on Set5
- **Training Time**: ~8-10 hours on RTX 3080

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

- [SRCNN Paper](https://arxiv.org/abs/1501.00092): "Image Super-Resolution Using Deep Convolutional Networks"
- [SRGAN Paper](https://arxiv.org/abs/1609.04802): "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
- [DIV2K Dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/): High-quality images for super-resolution
- [Set5 Dataset](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html): Standard super-resolution benchmark

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in `config.yaml`
2. **Dataset Not Found**: Ensure dataset paths are correct in configuration
3. **Import Errors**: Check that all dependencies are installed
4. **Training Slow**: Verify CUDA installation and GPU availability

### Performance Tips

- Use mixed precision training (enabled by default)
- Increase `num_workers` for faster data loading
- Use SSD storage for dataset to reduce I/O bottleneck
- Monitor GPU utilization with `nvidia-smi`
