#!/usr/bin/env python3
from pathlib import Path
from nn_cnn_simple_01_gem import CnnSimpleGem
from nn_unet_simple_01_gem import UNetSimpleGem
from nn_unet_diff_01_gem import UNetDiffGem, sample_diffusion
from prep_data import preprocess_data, plot_channel_interactive
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader



def main():

    # Define ptions data (should be in config file)
    num_epochs = 30
    batch_size = 10
    train_perc = 0.8
    opt_dec = 'upsConv3' #'convTransp', 'upsConv2', 'upsConv3', pixShuff
    test_ind_choice = 'even' #rand, rand_fix, even, middle, edge,

    hide_channels = 16
    k = 3
    d = [1,2]
    output_size = (36, 108)
    lr = 1e-3

    # -------------------------------------------------------------
    # MODEL SELECTION TOGGLE
    # Options: 'cnn_simple_gem', 'unet_simple_gem', 'diff_unet'
    # -------------------------------------------------------------
    model_type = 'diff_unet'

    if model_type == 'diff_unet':
        lr = lr * 0.1
        hide_channels = hide_channels * 4
        num_epochs = num_epochs * 3


    fnamex = 'x700'
    in_variables = ['S_yz', 'T_yz', 'runoff_yz']  # Replace with your variable names
    out_variables = ['lifw_xy']  # Replace with your variable names
    root_dir = '/Users/ivankova/Desktop/beb/rx'
    filepaths = sorted(Path(root_dir).rglob(f'data_annual_{fnamex}.nc'))

    # Prepare data
    print('-----Preprocessing and splitting data')

    data_in = preprocess_data(filepaths, in_variables, donanmask = False)
    data_out = preprocess_data(filepaths, out_variables, donanmask = True)

    data_tensor_in = torch.tensor(data_in, dtype=torch.float32)  # Shape: (N, C, H, W)
    data_tensor_out = torch.tensor(data_out, dtype=torch.float32)  # Shape: (N, C, H, W)

    # Quick Standardization (Mean 0, Std 1) for stable diffusion
    if model_type == 'diff_unet':
        in_mean = data_tensor_in.mean(dim=(0, 2, 3), keepdim=True)
        in_std = data_tensor_in.std(dim=(0, 2, 3), keepdim=True)
        out_mean = data_tensor_out.mean(dim=(0, 2, 3), keepdim=True)
        out_std = data_tensor_out.std(dim=(0, 2, 3), keepdim=True)
        
        data_tensor_in = (data_tensor_in - in_mean) / (in_std + 1e-6)
        data_tensor_out = (data_tensor_out - out_mean) / (out_std + 1e-6)

    if test_ind_choice == 'even':  # rand, even, middle, edge,
        all_indices = np.arange(0, 50)
        test_indices = np.array([1, 7, 13, 16, 24, 31, 32, 39, 44, 50]) - 1
        train_indices = np.setdiff1d(all_indices, test_indices)
        #train_indices = np.random.shuffle(train_indices)
    elif test_ind_choice == 'middle':
        all_indices = np.arange(0, 50)
        test_indices = np.array([16, 17, 18, 19, 20, 21, 22, 23, 24, 25]) - 1
        train_indices = np.setdiff1d(all_indices, test_indices)
    elif test_ind_choice == 'rand':
        indices = np.random.permutation(len(data_tensor_in))
        train_size = int(train_perc * len(data_tensor_in))
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
    elif test_ind_choice == 'rand_fix':
        all_indices = np.arange(0, 50)
        test_indices = np.array([28, 4, 14, 17, 26, 38, 35, 29,  5, 42]) - 1
        train_indices = np.setdiff1d(all_indices, test_indices)

    np.random.shuffle(train_indices)

    train_in, test_in = data_tensor_in[train_indices], data_tensor_in[test_indices]
    train_out, test_out = data_tensor_out[train_indices], data_tensor_out[test_indices]

    in_channels = train_in.shape[1]
    out_channels = train_out.shape[1]

    print(data_tensor_in.shape)
    print(train_in.shape)
    print(test_in.shape)

    print(f'Number of input channels = {in_channels}')
    print(f'Number of output channels = {out_channels}')

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(train_in, train_out)
    test_dataset = TensorDataset(test_in, test_out)

    # shuffle=True ensures the data is randomized at the start of EVERY epoch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define nn options
    print(f'-----Defining model: {model_type}')

    if model_type == 'cnn_simple_gem':
        model = CnnSimpleGem(in_channels=in_channels, out_channels=out_channels, hide_channels=hide_channels, k=k, d=d)
    elif model_type == 'unet_simple_gem':
        model = UNetSimpleGem(in_channels=in_channels, out_channels=out_channels, hide_channels=hide_channels, k=k, d=d, out_size=output_size)
    elif model_type == 'diff_unet':
        model = UNetDiffGem(in_channels=in_channels, out_channels=out_channels, hide_channels=hide_channels, k=k, d=d, out_size=output_size)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Options: 'cnn_simple_gem', 'unet_simple_gem', 'diff_unet'")

    # Check if it loads correctly
    print(model)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Setup diffusion parameters if model_type is 'diff_unet'
    if model_type == 'diff_unet':
        T = 1000
        beta = torch.linspace(1e-4, 0.02, T)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

    # Train network
    print('-----Training network')

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()  

        # Iterate directly over the train_loader
        for inputs, labels in train_loader:
            optimizer.zero_grad()  
            
            if model_type == 'diff_unet':
                t = torch.randint(0, T, (inputs.size(0),), device=labels.device)
                noise = torch.randn_like(labels)
                a_bar_t = alpha_bar.to(labels.device)[t].view(-1, 1, 1, 1)
                y_t = torch.sqrt(a_bar_t) * labels + torch.sqrt(1.0 - a_bar_t) * noise
                
                outputs = model(y_t, inputs, t)
                loss = criterion(outputs, noise) 
                
                loss.backward()  
                # ADD GRADIENT CLIPPING FOR STABILITY
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()  

                # Log the actual noise loss (smooth and stable)
                running_loss += loss.item() * inputs.size(0)

            else:
                outputs = model(inputs, opt_dec)
                loss = criterion(outputs, labels)
                
                loss.backward()  
                optimizer.step()  

                running_loss += loss.item() * inputs.size(0)

        avg_loss = running_loss / len(train_in)

        # Check validation loss
        val_loss = 0.0
        model.eval()  
        with torch.no_grad():
            # Iterate directly over the test_loader
            for inputs, labels in test_loader:
                if model_type == 'diff_unet':
                    t = torch.randint(0, T, (inputs.size(0),), device=labels.device)
                    noise = torch.randn_like(labels)
                    a_bar_t = alpha_bar[t].view(-1, 1, 1, 1)
                    y_t = torch.sqrt(a_bar_t) * labels + torch.sqrt(1.0 - a_bar_t) * noise
                    outputs = model(y_t, inputs, t)
                    
                    loss = criterion(outputs, noise)
                else:
                    outputs = model(inputs, opt_dec)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)

            val_loss = val_loss / len(test_in)
            print(f'Epoch {epoch + 1:>3}, Average Loss: {avg_loss:>10.6f}, Validation Loss: {val_loss:>10.6f}')

    # Evaluate network's performance
    print('-----Evaluating network: plots')

    model.eval()
    num_samples_to_plot = 5  # You can change this to plot more or fewer samples

    with torch.no_grad():
        # Predict lifw_xy ONLY for the samples we intend to plot (prevents OOM)
        if model_type == 'diff_unet':
            print("Sampling reverse diffusion process for test/train predictions...")
            predicted_train = sample_diffusion(model, train_in[:num_samples_to_plot], out_channels, output_size, T, alpha, alpha_bar, beta, num_steps=50)
            predicted_test = sample_diffusion(model, test_in[:num_samples_to_plot], out_channels, output_size, T, alpha, alpha_bar, beta, num_steps=50)
            
            # UN-NORMALIZE BACK TO PHYSICAL UNITS FOR DIFFUSION PLOTTING
            predicted_train = (predicted_train * out_std.to(predicted_train.device)) + out_mean.to(predicted_train.device)
            predicted_test = (predicted_test * out_std.to(predicted_test.device)) + out_mean.to(predicted_test.device)
            plot_train_out = (train_out[:num_samples_to_plot] * out_std.to(train_out.device)) + out_mean.to(train_out.device)
            plot_test_out = (test_out[:num_samples_to_plot] * out_std.to(test_out.device)) + out_mean.to(test_out.device)
        else:
            predicted_train = model(train_in[:num_samples_to_plot], opt_dec)
            predicted_test = model(test_in[:num_samples_to_plot], opt_dec)
            # Standard models don't use standardization in this script, so we just pass data through
            plot_train_out = train_out[:num_samples_to_plot]
            plot_test_out = test_out[:num_samples_to_plot]

    # Collect all data to find global min and max (safely convert to numpy first)
    all_data = np.concatenate([
        plot_train_out[:, 0, :, :].cpu().numpy().flatten(),
        predicted_train[:, 0, :, :].cpu().numpy().flatten(),
        plot_test_out[:, 0, :, :].cpu().numpy().flatten(),
        predicted_test[:, 0, :, :].cpu().numpy().flatten()
    ])

    vmin, vmax = np.min(all_data), np.max(all_data)
    vmax = vmax / 2

    for ttest in range(2):
        # Create a figure with num_samples_to_plot rows and 2 columns for actual and predicted plots
        fig, axes = plt.subplots(num_samples_to_plot, 2, figsize=(10, 10))

        for i in range(num_samples_to_plot):
            if ttest == 0:
                dat_actual = plot_train_out[i, 0, :, :].cpu().numpy()
                dat_pred = predicted_train[i, 0, :, :].cpu().numpy()
                ttl = 'Train'
            else:
                dat_actual = plot_test_out[i, 0, :, :].cpu().numpy()
                dat_pred = predicted_test[i, 0, :, :].cpu().numpy()
                ttl = 'Test'

            # Plot actual lifw_xy
            im_actual = axes[i, 0].imshow(dat_actual, cmap='magma_r', vmin=vmin, vmax=vmax)
            axes[i, 0].set_title(f'{ttl}: Actual (Sample {i + 1})')
            fig.colorbar(im_actual, ax=axes[i, 0])

            # Plot predicted lifw_xy
            im_pred = axes[i, 1].imshow(dat_pred, cmap='magma_r', vmin=vmin, vmax=vmax)
            axes[i, 1].set_title(f'{ttl}: Predicted (Sample {i + 1})')
            fig.colorbar(im_pred, ax=axes[i, 1])

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()