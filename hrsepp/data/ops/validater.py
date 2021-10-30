def _check_x(inputs):

    # check inputs by following constrains
    # 1) if any feature of inputs is all NaN, then discard this inputs.
    # 2) if dimensions of inputs is not equal to 4, then raise error.

    if inputs.ndim == 4:
        Nt, Nlat, Nlon, Nf = inputs.shape

        #for i in range(Nf):
        pass
    else:
        raise TypeError('The dimension is not equal to 4')
