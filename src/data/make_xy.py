import numpy as np



def make_input(inputs, 
               outputs,
               len_input, 
               len_output, 
               window_size):
    """Generate inputs and outputs for LSTM."""
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
    
    return inputs, np.squeeze(outputs)