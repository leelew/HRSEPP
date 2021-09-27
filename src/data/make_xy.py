# ==============================================================================
# Make input for single grid or multi-grids (images)
#
# author: Lu Li, 2021/09/27
# ============================================================================== 

import numpy as np


def make_input(inputs, 
               outputs,
               len_input, 
               len_output, 
               window_size,
               use_lag_y=True):
    """Generate inputs and outputs for LSTM."""

    if use_lag_y:
        inputs = np.concatenate([inputs, outputs], axis=-1)

    # caculate the last time point to generate batch
    end_idx = inputs.shape[0] - len_input - len_output - window_size

    # generate index of batch start point in order
    batch_start_idx = range(end_idx)

    # get batch_size
    batch_size = len(batch_start_idx)

    # generate inputs
    input_batch_idx = [
        (range(i, i + len_input)) for i in batch_start_idx]
    inputs = np.take(inputs, input_batch_idx, axis=0). \
        reshape(batch_size, len_input,
                inputs.shape[1])

    # generate outputs
    output_batch_idx = [
        (range(i + len_input + window_size, i + len_input + window_size +
               len_output)) for i in batch_start_idx]
    outputs = np.take(outputs, output_batch_idx, axis=0). \
        reshape(batch_size,  len_output,
                outputs.shape[1])
    
    return inputs, outputs


def make_image_inputs(X, 
               y,
               len_input, 
               len_output, 
               window_size,
               use_lag_y=True):
               
    if use_lag_y:
        Nf = X.shape[-1] + 1
    else:
        Nf = X.shape[-1]

    N_sample = X.shape[0] - len_input - window_size - len_output

    X_f = np.full((N_sample, len_input, X.shape[1], X.shape[2], Nf), np.nan)
    y_f = np.full((N_sample, len_output, y.shape[1], y.shape[2], 1), np.nan)

    for i in np.arange(X.shape[1]):
        for j in np.arange(X.shape[2]):
            X_f[:, :, i,j, :], y_f[:, :, i,j, :] = make_input(
                                X[:, i, j, :], 
                                y[:, i, j, :], 
                                len_input=len_input, 
                                len_output=len_output, 
                                window_size=window_size, 
                                use_lag_y=use_lag_y)
                                
    return X_f, y_f
