import numpy as np

def get_im2col_indices(inputs_shape, h_kernel, w_kernel, padding, stride):
    '''
    Im2Col Helper Function
    
    :param inputs_shape: (int tuple) shape of the input tensor in the form (N, C, H, W) where N is 
        the batch size, C is the number of channels and (H, W) is the height and width
    
    :param h_kernel: (int) height of the convolutional kernel
    :param w_kernel: (int) width of the convolutional kernel
    :param padding: (int tuple) padding applied to the input in the form (padX, padY)
    :param stride: (int tuple) stride length of the convolution in the form (strideX, strideY)
    
    :returns (k, i, j) representing the indices needed for im2col
    
    '''
    N, C, H, W = inputs_shape
    h_pad, w_pad = padding
    h_stride, w_stride = stride

    out_height = int(1 + (H - h_kernel + 2 * h_pad) // h_stride)
    out_width = int(1 + (W - w_kernel + 2 * w_pad) // w_stride)

    i0 = np.repeat(np.arange(h_kernel), w_kernel)
    i0 = np.tile(i0, C)
    i1 = h_stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(w_kernel), h_kernel * C)
    j1 = w_stride * np.tile(np.arange(out_width), out_height)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), h_kernel * w_kernel).reshape(-1, 1)

    return (k, i, j)

def im2col_indices(inputs, h_kernel, w_kernel, padding, stride):
    '''
    Numpy implementation of the Im2Col Algorithm
    
    Source
        - Original Paper: https://inria.hal.science/inria-00112631/file/p1038112283956.pdf
        - Implementation modified from Stanford CS231 GitHub
    
    :param inputs: (numpy.ndarray) input tensor with shape (N, C, H, W) where N is 
        the batch size, C is the number of channels and (H, W) is the height and width
    
    :param h_kernel: (int) height of the convolutional kernel
    :param w_kernel: (int) width of the convolutional kernel
    :param padding: (int tuple) padding applied to the input in the form (padX, padY)
    :param stride: (int tuple) stride length of the convolution in the form (strideX, strideY)
    
    :returns numpy.ndarray of shape (h_kernel * w_kernel * C, N), which is the column 
        representation of the convolved output
    
    '''
    h_pad, w_pad = padding
    h_stride, w_stride = stride

    inputs_padded = np.pad(inputs, ((0, 0), (0, 0), (h_pad, h_pad), (w_pad, w_pad)), mode = 'constant')

    k, i, j = get_im2col_indices(inputs.shape, h_kernel, w_kernel, padding, stride)

    cols = inputs_padded[:, k, i, j]
    cols = cols.transpose(1, 2, 0).reshape(h_kernel * w_kernel * inputs.shape[1], -1)

    return cols

def col2im_indices(cols, inputs_shape, h_kernel, w_kernel, padding, stride):
    '''
    Numpy implementation of the Col2Im Algorithm
    
    Source
        - Original Paper: https://inria.hal.science/inria-00112631/file/p1038112283956.pdf
        - Implementation modified from Stanford CS231 GitHub
    
    :param cols: (numpy.ndarray) column representation of convolved output
    :param inputs_shape: (int tuple) shape of the input tensor in the form (N, C, H, W) where N is 
        the batch size, C is the number of channels and (H, W) is the height and width
    
    :param h_kernel: (int) height of the convolutional kernel
    :param w_kernel: (int) width of the convolutional kernel
    :param padding: (int tuple) padding applied to the input in the form (padX, padY)
    :param stride: (int tuple) stride length of the convolution in the form (strideX, strideY)
    
    :returns numpy.ndarray of shape (N, C, H, W), which is the image representation of the column input
    
    '''
    N, C, H, W = inputs_shape
    h_pad, w_pad = padding

    H_padded, W_padded = H + 2 * h_pad, W + 2 * w_pad
    inputs_padded = np.zeros((N, C, H_padded, W_padded), dtype = cols.dtype)

    k, i, j = get_im2col_indices(inputs_shape, h_kernel, w_kernel, padding, stride)

    cols_reshaped = cols.reshape(C * h_kernel * w_kernel, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(inputs_padded, (slice(None), k, i, j), cols_reshaped)

    if padding == (0, 0):
        return inputs_padded

    return inputs_padded[:, :, h_pad : -h_pad, w_pad : -w_pad]
