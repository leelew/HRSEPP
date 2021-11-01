from data.ops.make_X_y import make_Xy


class DataLoader():
    #[(sample, in_len, lat, lon, f), (sample, out_len, lat, lon, 1)]

    def __init__(self,
                 len_input,
                 len_output,
                 window_size,
                 use_lag_y=True,
                 mode='train'):
        self.len_input = len_input
        self.len_output = len_output
        self.window_size = window_size
        self.use_lag_y = use_lag_y

    def __call__(self, X, y):
        # generate inputs
        X, y = make_Xy(X, y, self.len_input, self.len_output, self.window_size,
                       self.use_lag_y)

        #TODO:Add choice for different case

        return X, y
