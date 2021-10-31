import numpy as np


class MakeXy():
    """[(t, f, lat, lon), (t, 1, lat, lon)] make to 
       [(sample, in_len, lat, lon, f), (sample, out_len, lat, lon, n_out)]
    """
    def __init__(self, len_input, len_output, aux, window_size, mode='train'):
        pass

    def __call__(self, X):
        pass

    def make_grid_train_xy(
        inputs,  #(t, f, lat, lon)
        outputs,
        len_input,
        len_output,
        window_size,
    ):

        Nt, Nf, Nlat, Nlon = inputs.shape
        """Generate inputs and outputs for LSTM."""
        # caculate the last time point to generate batch
        end_idx = inputs.shape[0] - len_input - len_output - window_size

        # generate index of batch start point in order
        batch_start_idx = range(end_idx)

        # get batch_size
        batch_size = len(batch_start_idx)

        # generate inputs
        input_batch_idx = [(range(i, i + len_input)) for i in batch_start_idx]
        inputs = np.take(inputs, input_batch_idx,
                         axis=0).reshape(batch_size, len_input, Nf, Nlat, Nlon)

        # generate outputs
        output_batch_idx = [(range(i + len_input + window_size,
                                   i + len_input + window_size + len_output))
                            for i in batch_start_idx]
        outputs = np.take(outputs, output_batch_idx, axis=0). \
            reshape(batch_size,  len_output, 1, Nlat, Nlon)

        return inputs, outputs