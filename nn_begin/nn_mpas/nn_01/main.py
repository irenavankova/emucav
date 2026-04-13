#!/usr/bin/env python3
from pathlib import Path
#from nn_cnn_simple import CnnSimple
#from nn_cnn_simple_gem import CnnSimple
#from nn_cnn_simple_01 import CnnSimple
from nn_cnn_simple_01_gem import CnnSimpleGem
from prep_data import preprocess_data, plot_channel_interactive
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def main():

    # Define ptions data (should be in config file)
    num_epochs = 10
    batch_size = 10
    train_perc = 0.8
    opt_dec = 'upsConv3' #'convTransp', 'upsConv2', 'upsConv3', pixShuff
    test_ind_choice = 'even' #rand, rand_fix, even, middle, edge,

    hide_channels = 16
    k = 3
    d = [1,2]

    #ttest = 1
    #train_ind
    #validate_ind


    fnamex = 'x700'
    in_variables = ['S_yz', 'T_yz', 'runoff_yz']  # Replace with your variable names
    out_variables = ['lifw_xy']  # Replace with your variable names
    root_dir = '/Users/irenavankova/Desktop/beb/rx'
    filepaths = sorted(Path(root_dir).rglob(f'data_annual_{fnamex}.nc'))

    # Prepare data
    print('-----Preprocessing and splitting data')

    data_in = preprocess_data(filepaths, in_variables, donanmask = False)
    data_out = preprocess_data(filepaths, out_variables, donanmask = True)

    data_tensor_in = torch.tensor(data_in, dtype=torch.float32)  # Shape: (N, C, H, W)
    data_tensor_out = torch.tensor(data_out, dtype=torch.float32)  # Shape: (N, C, H, W)

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

    # Define nn options
    print('-----Defining model')

    # Model, loss, optimizer
    #model = CnnSimple(in_channels=in_channels, out_channels=out_channels)
    model = CnnSimpleGem(in_channels=in_channels, out_channels=out_channels, hide_channels = hide_channels, k=k, d=d)


    # Check if it loads correctly
    print(model)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # Train network
    print('-----Training network')

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()  # Set model to training mode

        for j in range(0, len(train_in), batch_size):
            optimizer.zero_grad()  # Zero gradients at the start of each mini-batch

            # Iterate over the next 10 images
            for i in range(j, min(j + 10, len(train_in))):
                inputs = train_in[i].unsqueeze(0)  # Shape: (1, C, H, W)
                labels = train_out[i].unsqueeze(0)  # Shape: (1, C, H, W)

                outputs = model(inputs, opt_dec)

                loss = criterion(outputs, labels)
                loss.backward()  # Accumulate gradients

                running_loss += loss.item()
                #print(f'Instantaneous Loss: {loss.item()}')

            optimizer.step()  # Update weights after every 10 images

        # Print statistics
        avg_loss = running_loss / len(train_in)
        #print(f'Epoch {epoch + 1}, Average Loss: {avg_loss}')

        #Check validation loss
        val_loss = 0.0
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for j in range(0, len(test_in), batch_size):
                optimizer.zero_grad()  # Zero gradients at the start of each mini-batch

                # Iterate over the next 10 images
                for i in range(j, min(j + 10, len(test_in))):
                    inputs = test_in[i].unsqueeze(0)  # Shape: (1, C, H, W)
                    labels = test_out[i].unsqueeze(0)  # Shape: (1, C, H, W)

                    outputs = model(inputs, opt_dec)

                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    # print(f'Instantaneous Loss: {loss.item()}')

                optimizer.step()  # Update weights after every 10 images

            # Print statistics
            val_loss = val_loss / len(test_in)
            #print(f'Epoch {epoch + 1}, Validation Loss: {val_loss}')
            #print(f'Epoch {epoch + 1}, Average Loss: {avg_loss}, Validation Loss: {val_loss}')
            print(f'Epoch {epoch + 1:>3}, Average Loss: {avg_loss:>10.6f}, Validation Loss: {val_loss:>10.6f}')

    # Evaluate network's performance
    print('-----Evaluating network: plots')

    # After training, evaluate the model on the test dataset and plot predictions
    model.eval()
    with torch.no_grad():
        # Predict lifw_xy for the test dataset
        predicted_train = model(train_in, opt_dec)
        predicted_test = model(test_in, opt_dec)

    # Plot predicted vs actual lifw_xy for a few samples

    num_samples_to_plot = 5  # You can change this to plot more or fewer samples

    # Collect all data to find global min and max
    all_data = np.concatenate([
        train_out[:num_samples_to_plot, 0, :, :].flatten(),
        predicted_train[:num_samples_to_plot, 0, :, :].flatten(),
        test_out[:num_samples_to_plot, 0, :, :].flatten(),
        predicted_test[:num_samples_to_plot, 0, :, :].flatten()
    ])

    vmin, vmax = np.min(all_data), np.max(all_data)
    vmax = vmax/2

    for ttest in range(2):
        # Create a figure with 1 row and 5 columns for actual and predicted plots
        fig, axes = plt.subplots(num_samples_to_plot, 2, figsize=(10, 10))

        for i in range(num_samples_to_plot):
            if ttest == 0:
                dat_actual = train_out[i, 0, :, :]
                dat_pred = predicted_train[i, 0, :, :]
                ttl = 'Train'
            else:
                dat_actual = test_out[i, 0, :, :]
                dat_pred = predicted_test[i, 0, :, :]
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
