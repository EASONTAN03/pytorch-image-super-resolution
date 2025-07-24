import os
import shutil
import time
import csv
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from datetime import datetime
from utils.metrics import PSNR, SSIM
from utils.models import SRCNN, SRGAN 
from utils.utils import *
from utils.prepare_data import prepare_data


MODEL_MAP = {
    "SRCNN": SRCNN.train, # Assuming SRCNN.train is a static/class method for training
    "SRGAN": SRGAN.train, # Add SRGAN.train when implemented
}

# --- Main Training Orchestration Function ---
def main() -> None:
    # Load configuration
    config = load_config("../config.yaml") # Ensure path to config.yaml is correct
    train_config = config.get("train", {})

    # Set all random seeds for reproducibility
    set_all_seeds(config['SEED'])

    # Create base paths for saving models, logs, etc.
    # The create_base_path function should return a PathLike object or similar for directories
    base_dirs = create_base_path() # e.g., returns {'models': Path('models_dir'), 'logs': Path('logs_dir')}
    
    # Prepare data and get the loaders
    data_info = prepare_data()
    train_loader = data_info["train_loader"]
    valid_loader = data_info["valid_loader"]
    scale_factor = data_info["scale_factor"] # Not directly used in main, but good to know it's available

    # Get current timestamp for unique saving directories and logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_name=train_config['model']
    model_dir_name = f"{model_name}_{train_config['benchmark']}_{timestamp}"
    model_dir = base_dirs['models'] / model_dir_name
    os.makedirs(model_dir, exist_ok=True) # Create the directory if it doesn't exist
    save_dir={
        "models": model_dir,
        "resume": base_dirs["models"] / train_config[model_name].get("resume") if train_config[model_name].get("resume") != "" else None,
        "log_path": model_dir / "log.json",
        "results_path": base_dirs["results"] / "model_results.csv"
    }
    print(f"Output models to: {save_dir}")

    # --- SRCNN Training Branch ---
    if model_name == "SRCNN":
        print("Starting SRCNN training branch.")
        
        # The 'train' function now expects DataLoaders directly and passes SRCNN_config
        model, training_logs = MODEL_MAP["SRCNN"](
            train_loader=train_loader,
            val_loader=valid_loader,
            SRCNN_config=train_config["SRCNN"], # Pass the SRCNN specific config
            save_dir=save_dir,
            SEED=config["SEED"] # Pass the seed for reproducibility
        )

        print("SRCNN training complete.")

        # --- Extract training logs for saving ---
        train_losses = training_logs['train_losses']
        val_losses = training_logs['val_losses']
        val_psnr_scores = training_logs['val_psnr_scores']
        val_ssim_scores = training_logs['val_ssim_scores']
        
        # Get best validation results from logs
        best_val_psnr = max(val_psnr_scores) if val_psnr_scores else 0.0
        best_val_ssim = max(val_ssim_scores) if val_ssim_scores else 0.0

        # Find the index of the best PSNR to get corresponding loss
        best_psnr_idx = val_psnr_scores.index(best_val_psnr) if val_psnr_scores else -1
        best_val_loss_at_best_psnr = val_losses[best_psnr_idx] if best_psnr_idx != -1 else float('inf')

        # --- Save Training Graphs ---
        save_training_graph(train_losses, val_losses, "Training and Validation Loss", "Loss", save_dir["models"] / "loss_graph.png")
        save_training_graph(val_psnr_scores, val_psnr_scores, "Validation PSNR", "PSNR", save_dir["models"] / "val_psnr_graph.png") # PSNR for both train and val usually same for validation
        save_training_graph(val_ssim_scores, val_ssim_scores, "Validation SSIM", "SSIM", save_dir["models"] / "val_ssim_graph.png") # SSIM for both train and val usually same for validation

        # --- Save Model Checkpoint ---
        model_save_path = save_dir["models"] / f"{train_config['model']}_model.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to: {model_save_path}")

        # --- Prepare Log Data for CSV ---
        log_data = {
            "start_epoch": training_logs["start_epoch"],
            "epochs_run": training_logs["epochs_run"],
            "model_name": model_dir_name,
            "dataset_config": config["dataset"],
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_loss": val_losses[-1] if val_losses else None,
            "best_val_psnr": best_val_psnr,
            "best_val_ssim": best_val_ssim,
            "best_val_loss_at_best_psnr": best_val_loss_at_best_psnr,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_psnr_scores": val_psnr_scores,
            "val_ssim_scores": val_ssim_scores,
            "train_config": train_config["SRCNN"], # Include the SRCNN config used
        }
    # ---  SRGAN Training Branch ---
    elif model_name == "SRGAN":
        print("Starting SRGAN training branch.")

        model, training_logs = MODEL_MAP["SRGAN"](
            train_loader=train_loader,
            valid_loader=valid_loader,
            SRGAN_config=train_config["SRGAN"],
            save_dir=save_dir,
            SEED=config["SEED"]
        )

        print("SRGAN training complete.")

        train_losses_generator = training_logs['train_losses_generator']
        train_losses_discriminator = training_logs['train_losses_discriminator']
        val_losses_generator = training_logs['val_losses_generator']
        val_psnr_scores = training_logs['val_psnr_scores']
        val_ssim_scores = training_logs['val_ssim_scores']

        best_val_psnr = max(val_psnr_scores) if val_psnr_scores else 0.0
        best_val_ssim = max(val_ssim_scores) if val_ssim_scores else 0.0

        best_psnr_idx = val_psnr_scores.index(best_val_psnr) if val_psnr_scores else -1
        best_val_loss_at_best_psnr = val_losses_generator[best_psnr_idx] if best_psnr_idx != -1 else float('inf')


        save_training_graph(train_losses_generator, val_losses_generator, "Training and Validation G Loss", "Loss", save_dir["models"] / "g_loss_graph.png")
        save_training_graph(train_losses_discriminator, train_losses_discriminator, "Training D Loss", "Loss", save_dir["models"] / "d_loss_graph.png")
        save_training_graph(val_psnr_scores, val_psnr_scores, "Validation PSNR", "PSNR", save_dir["models"] / "val_psnr_graph.png") # PSNR for both train and val usually same for validation
        save_training_graph(val_ssim_scores, val_ssim_scores, "Validation SSIM", "SSIM", save_dir["models"] / "val_ssim_graph.png") # SSIM for both train and val usually same for validation

        torch.save(model.generator.state_dict(), os.path.join(save_dir["models"], "SRGAN_generator.pth"))
        torch.save(model.discriminator.state_dict(), os.path.join(save_dir["models"], "SRGAN_discriminator.pth"))
        print(f"Generator and Discriminator models saved to: {save_dir['models']}")

        # Save final log data to JSON file
        log_data = {
            "start_epoch": training_logs['start_epoch'],
            "epochs_run": training_logs['epochs_run'],
            "model_name": model_dir_name,
            "dataset_config": config["dataset"],
            "final_train_g_loss": train_losses_generator[-1] if train_losses_generator else None,
            "final_train_d_loss": train_losses_discriminator[-1] if train_losses_discriminator else None,
            "final_val_g_loss": val_losses_generator[-1] if val_losses_generator else None,
            "final_val_psnr": val_psnr_scores[-1] if val_psnr_scores else None,
            "final_val_ssim": val_ssim_scores[-1] if val_ssim_scores else None,
            "best_val_psnr": best_val_psnr,
            "best_val_ssim": best_val_ssim,
            "best_val_loss_at_best_psnr": best_val_loss_at_best_psnr,
            "train_losses_generator": train_losses_generator,
            "train_losses_discriminator": train_losses_discriminator,
            "val_losses_generator": val_losses_generator,
            "val_psnr_scores": val_psnr_scores,
            "val_ssim_scores": val_ssim_scores,
            "train_config": train_config["SRGAN"],
        }

        
    else:
        print(f"Model '{config['model']}' not recognized or implemented.")

    save_log(save_dir["log_path"], log_data)
    print(f"Saved training log to {save_dir['log_path']}")

    csv_row = {
            "model_dir_name": model_dir_name,
            "model_type": train_config["model"],
            "scaler_factor": scale_factor,
            "best_psnr": best_val_psnr,
            "best_ssim": best_val_ssim
        }
    
    write_header = not save_dir["results_path"].exists()
    with open(save_dir["results_path"], mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=csv_row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(csv_row)
    print(f"Appended results to {save_dir['results_path']}")

if __name__ == "__main__":
    main()