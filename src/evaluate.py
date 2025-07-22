    if X_test_tensor is not None and y_test_tensor is not None:
        print("Evaluating on test set...")
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # Batch size 1 for evaluation if needed

        model.eval()
        test_running_loss = 0.0
        with torch.no_grad():
            for lr_images_test, hr_images_test in test_loader:
                lr_images_test = lr_images_test.to(device)
                hr_images_test = hr_images_test.to(device)
                
                outputs_test = model(lr_images_test)
                test_loss = criterion(outputs_test, hr_images_test)
                test_running_loss += test_loss.item() * lr_images_test.size(0)
        
        avg_test_loss = test_running_loss / len(test_loader.dataset)
        print(f"Test Loss: {avg_test_loss:.6f}")


def validate(model: nn.Module,
             data_dataloader: DataLoader,
             epoch: int,
             writer: SummaryWriter,
             psnr_model: nn.Module,
             ssim_model: nn.Module) -> [float, float]:
    """Test main program

    Args:
        model (nn.Module): generator model in adversarial networks
        data_prefetcher (CUDAPrefetcher): test dataset iterator
        epoch (int): number of test epochs during training of the adversarial network
        writer (SummaryWriter): log file management function
        psnr_model (nn.Module): The model used to calculate the PSNR function
        ssim_model (nn.Module): The model used to compute the SSIM function
        mode (str): test validation dataset accuracy or test dataset accuracy

    """
    # Calculate how many batches of data are in each Epoch
    batches = len(data_dataloader)
    batch_time = AverageMeter("Time", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    ssimes = AverageMeter("SSIM", ":4.4f")
    progress = ProgressMeter(len(data_dataloader), [batch_time, psnres, ssimes], prefix=f"{mode}: ")

    # Put the adversarial network model in validation mode
    model.eval()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    data_dataloader.reset()
    batch_data = data_dataloader.next()

    # Get the initialization test time
    end = time.time()

    with torch.no_grad():
        while batch_data is not None:
            # Transfer the in-memory data to the CUDA device to speed up the test
            lr = batch_data["lr"].to(device=config["device"], memory_format=torch.channels_last, non_blocking=True)
            hr = batch_data["hr"].to(device=config["device"], memory_format=torch.channels_last, non_blocking=True)

            # Use the generator model to generate a fake sample
            with amp.autocast():
                sr = model(lr)

            # Statistical loss value for terminal data output
            psnr = psnr_model(sr, hr)
            ssim = ssim_model(sr, hr)
            psnres.update(psnr.item(), lr.size(0))
            ssimes.update(ssim.item(), lr.size(0))

            # Calculate the time it takes to fully test a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Record training log information
            if batch_index % (batches // 5) == 0:
                progress.display(batch_index)

            # Preload the next batch of data
            batch_data = data_dataloader.next()

            # After training a batch of data, add 1 to the number of data batches to ensure that the
            # terminal print data normally
            batch_index += 1

        # print metrics
        progress.display_summary()
    
        writer.add_scalar(f"Valid/PSNR", psnres.avg, epoch + 1)
        writer.add_scalar(f"Valid/SSIM", ssimes.avg, epoch + 1)

        return psnres.avg, ssimes.avg


if X_test_tensor is not None and y_test_tensor is not None:
        print("Evaluating on test set...")
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False) # Use a suitable batch size for test

        model.eval()
        test_running_loss = 0.0
        test_psnr_sum = 0.0
        test_ssim_sum = 0.0
        with torch.no_grad():
            for lr_images_test, hr_images_test in test_loader:
                lr_images_test = lr_images_test.to(device, non_blocking=True)
                hr_images_test = hr_images_test.to(device, non_blocking=True)
                
                outputs_test = model(lr_images_test)
                test_loss = criterion(outputs_test, hr_images_test)
                test_running_loss += test_loss.item() * lr_images_test.size(0)

                test_psnr_sum += psnr_metric(outputs_test, hr_images_test).item() * lr_images_test.size(0)
                test_ssim_sum += ssim_metric(outputs_test, hr_images_test).item() * lr_images_test.size(0)
        
        avg_test_loss = test_running_loss / len(test_loader.dataset)
        avg_test_psnr = test_psnr_sum / len(test_loader.dataset)
        avg_test_ssim = test_ssim_sum / len(test_loader.dataset)
        print(f"Test Loss: {avg_test_loss:.6f}, Test PSNR: {avg_test_psnr:.4f}, Test SSIM: {avg_test_ssim:.4f}")
