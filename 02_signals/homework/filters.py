import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    for i in range(Hi):
        for j in range(Wi):
            conv = 0
            for k in range(Hk):
                for l in range(Wk):
                    ko = k - Hk // 2
                    lo = l - Wk // 2
                    val = 0 if (not 0 <= i + ko < Hi) or (not 0 <= j+lo < Wi) else image[i+ko, j+lo]
                    conv += val * kernel[Hk - k - 1,Wk - l - 1]
            out[i,j]=conv
    
    out -= out.min()
    out *= 255 / out.max()
    out = out.astype(image.dtype)
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = np.zeros_like(image)

    ### YOUR CODE HERE
    out = np.zeros((H+pad_height*2, W+pad_width*2), dtype=image.dtype)
    out[pad_height:H+pad_height, pad_width:W+pad_width] = image.copy()
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    kernel = np.flip(kernel,0)
    kernel = np.flip(kernel,1)
    
    H_pad = Hk // 2
    W_pad = Wk // 2
    image = zero_pad(image, H_pad, W_pad)

    for i in range(Hi):
        for j in range(Wi):
            out[i, j] = np.sum(image[i:i+Hk,j:j+Wk]*kernel)
    
    out -= out.min()
    out *= 255 / out.max()
    out = out.astype(image.dtype)
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    image_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(kernel, s=image.shape)
    out = np.fft.ifft2(image_fft * kernel_fft)

    out -= out.min()
    out *= 255 / out.max()
    out = np.abs(out)
    out = out.astype(image.dtype)
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    ### YOUR CODE HERE
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    out = np.zeros((Hi, Wi))

    H_pad = Hk // 2
    W_pad = Wk // 2
    f = zero_pad(f, H_pad, W_pad)

    for i in range(Hi):
        for j in range(Wi):
            out[i, j] = np.sum(f[i:i+Hk,j:j+Wk]*g)
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = np.zeros_like(f)
    ### YOUR CODE HERE
    g = g - g.mean()
    out = cross_correlation(f,g)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = np.zeros_like(f)
    ### YOUR CODE HERE
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    out = np.zeros((Hi, Wi))

    H_pad = Hk // 2
    W_pad = Wk // 2
    f = zero_pad(f, H_pad, W_pad)

    g_mean = g.mean()
    s_g = np.std(g)
    g = (g-g_mean) / s_g

    for i in range(Hi):
        for j in range(Wi):
            patch = f[i:i+Hk, j:j+Wk]
            s_mean = np.std(patch)
            s_f = np.std(patch)

            out[i, j] = np.sum(((patch-s_mean)/s_f) * g)
    ### END YOUR CODE

    return out