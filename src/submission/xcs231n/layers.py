from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # ### START CODE HERE ###
    reshaped_x = x.reshape(x.shape[0], -1)  # Reshape x to (N, D)
    out = np.dot(reshaped_x, w) + b  # Compute the affine transformation
    # ### END CODE HERE ###
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # ### START CODE HERE ###
    reshaped_x = x.reshape(x.shape[0], -1)  # Reshape x to (N, D)
    # Gradient w.r.t. x
    dx = np.dot(dout, w.T).reshape(x.shape)  
    
    # Gradient w.r.t. w
    # Here reshaped_x dim: (N, D), dout dim: (N, M)
    # Hence reshaped_x.T dim: (D, N).dout(N, M) so dw dim: (D, M)
    dw = np.dot(reshaped_x.T, dout)
    
    # Gradient w.r.t. b     
    db = np.sum(dout, axis=0)
    # ### END CODE HERE ###
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # ### START CODE HERE ###
    out = np.maximum(0, x)
    # ### END CODE HERE ###
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # ### START CODE HERE ###
    dx = dout * (x > 0)
    # ### END CODE HERE ###
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # ### START CODE HERE ###
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    # ### END CODE HERE ###
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        pass
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # ### START CODE HERE ###
        running_mean = momentum * running_mean + (1 - momentum) * np.mean(x, axis=0)
        running_var = momentum * running_var + (1 - momentum) * np.var(x, axis=0)
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        x_normalized = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma * x_normalized +  beta
        cache = (x, x_normalized, sample_mean, sample_var, gamma, beta, eps)
        # ### END CODE HERE ###
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        pass
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # ### START CODE HERE ###
        x_normalized = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_normalized + beta
        cache = (x, x_normalized, running_mean, running_var, gamma, beta, eps)
        # ### END CODE HERE ###
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # ### START CODE HERE ###
    x, x_normalized, sample_mean, sample_var, gamma, beta, eps = cache
    N, D = dout.shape
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_normalized, axis=0)

    dx_normalized = dout * gamma
    dvar = np.sum(dx_normalized * (x - sample_mean) * -0.5 * (sample_var + eps) ** (-1.5), axis=0)
    dmean = np.sum(dx_normalized * -1 / np.sqrt(sample_var + eps), axis=0) + \
        dvar * np.sum(-2 * (x - sample_mean), axis=0) / N
    
    dx = dx_normalized / np.sqrt(sample_var + eps) + \
        dvar * 2 * (x - sample_mean) / N + dmean / N
    # ### END CODE HERE ###
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    # HINT: A derivation of this result is here:                              #
    # http://cs.stanford.edu/people/jcjohns/batchnorm.pdf                     #
    ###########################################################################
    # ### START CODE HERE ###
    x, x_normalized, sample_mean, sample_var, gamma, beta, eps = cache
    N, D = dout.shape
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_normalized, axis=0)
    
    v = sample_var + eps
    
    dx = gamma * (dout / v ** (1 / 2) - \
                           (1 / ((N * v ** (3 / 2)))) * (np.sum(dout * v, axis=0) + (x - sample_mean)
                                                                  * np.sum(dout * (x - sample_mean),axis=0) ) )
    # ### END CODE HERE ###
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # ### START CODE HERE ###
    sample_mean = np.mean(x, axis=1, keepdims=True)  # Mean over features
    sample_var = np.var(x, axis=1, keepdims=True)    # Variance over features
    x_normalized = (x - sample_mean) / np.sqrt(sample_var + eps)  # Normalize
    out = gamma * x_normalized + beta  # Scale and shift
    cache = (x, x_normalized, sample_mean, sample_var, gamma, beta, eps)
    # ### END CODE HERE ###
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # ### START CODE HERE ###
    x, x_normalized, sample_mean, sample_var, gamma, beta, eps = cache
    N, D = x.shape
    # Gradient w.r.t. beta
    dbeta = np.sum(dout, axis=0)
    # Gradient w.r.t. gamma
    dgamma = np.sum(dout * x_normalized, axis=0)
    # Gradient w.r.t. x
    dx = (1. / D) * gamma * (sample_var + eps) ** (-1. / 2.) * (
        D * dout - np.sum(dout, axis=1, keepdims=True)
        - (x - sample_mean) * (sample_var + eps) ** (-1.0) * np.sum(dout * (x - sample_mean), axis=1, keepdims=True)
    )
    # ### END CODE HERE ###
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        pass
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # ### START CODE HERE ###
        mask = (np.random.rand(*x.shape) < p) / p  # Create dropout mask
        out = x * mask  # Apply dropout mask
        # ### END CODE HERE ###
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        pass
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # ### START CODE HERE ###
        out = x  # No dropout during testing
        # ### END CODE HERE ###
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        pass
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # ### START CODE HERE ###
        dx = dout * mask  # Backpropagate through dropout mask
        # ### END CODE HERE ###
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # ### START CODE HERE ###
    # Extract dimensions and parameters from inputs
    N, C, H, W = x.shape            # Input dimensions
    F, _, HH, WW = w.shape          # Filter dimensions where HH and WW are height and width of filter
    stride = conv_param['stride']
    pad = conv_param['pad']
    
    # Calculate output dimensions
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride
    out = np.zeros((N, F, H_out, W_out))

    # Pad the input in order to preserve spatial dimensions after convolution
    x_padded = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')

    # Perform the convolution operation
    for point in range(N):  # Loop over each point
        for filter in range(F):  # loop over each filter
            for h_out_index in range(H_out):
                for w_out_index in range(W_out):
                    # Calculate top left corner of the current "slice"
                    h_start = h_out_index * stride
                    w_start = w_out_index * stride
                    h_end = h_start + HH
                    w_end = w_start + WW

                    receptive_field = x_padded[point, :, h_start:h_end, w_start:w_end]

                    #Perform convolution operation (element-wise multiplication and sum = Wx + b)
                    out[point, filter, h_out_index, w_out_index] = np.sum(receptive_field * w[filter]) + b[filter]
    # ### END CODE HERE ###
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # ### START CODE HERE ###
    """
    1. Unpack Cache and Initialize Gradients
    Unpack: Retrieve x, w, b, and conv_param from the cache tuple. Also extract stride, pad, and all dimensions (N, C, H, W, F, HH, WW, H', W').
    Initialize Gradients:
    db should be an array of zeros with shape (F,).
    dw should be an array of zeros with shape (F, C, HH, WW).
    dx should be an array of zeros with the same shape as the original input x (N, C, H, W).
    Pad Input x: Create x_padded using np.pad, just like in the forward pass. You will also need a padded version of dx called dx_padded (same shape as x_padded), which you will accumulate gradients into.
    
    2. Calculate db (Gradient with respect to bias)
    db is the easiest. Since each bias b[f] was added uniformly across its entire output feature map (F, H', W'), the gradient of the bias (db[f]) is simply the sum of all upstream gradients (dout) for that specific filter/feature map.
    Hint: Use np.sum(dout, axis=(0, 2, 3)) to sum over the N, H', and W' dimensions of dout.
    
    3. Calculate dw (Gradient with respect to weights)
    dw has the same shape as w (F, C, HH, WW).
    You need nested loops similar to the forward pass (over N, F, H', W').
    At each location in the loop:
    Extract the corresponding receptive_field from x_padded (same slicing logic as forward pass).
    The update to the weight gradient dw[f, c, hh, ww] is the value of the input receptive_field[c, hh, ww] multiplied by the upstream gradient value dout[n, f, h, w].
    Hint: Use accumulation: dw[f] += receptive_field * dout[n, f, h, w]. This accumulation happens across all N, H', and W' positions.
    
    4. Calculate dx (Gradient with respect to input x)
    This is the most complex part. We are backpropagating the gradients through the convolution operation itself.
    Again, use the same nested loops (over N, F, H', W').
    At each location:
    We need to distribute the upstream gradient dout[n, f, h, w] across the entire input region that the filter touched (the receptive field).
    The contribution to the input gradient is dx_padded[n, :, h_start:h_end, w_start:w_end] += w[f] * dout[n, f, h, w].
    Hint: You are effectively performing a deconvolution or transposed convolution operation here with the flipped filter weights.
    Final Step for dx: After accumulating all gradients into dx_padded, you must unpad it to match the original x shape.
    Hint: Slice dx_padded to remove the borders you added initially. If pad was 1, slice off 1 pixel from each side: dx = dx_padded[n, :, pad:H+pad, pad:W+pad] (adjust slicing for the N and C dimensions).
    """
    # Unpack: Retrieve x, w, b, and conv_param from the cache tuple. Also extract stride, pad, and all dimensions (N, C, H, W, F, HH, WW, H', W').
    (x, w, b, conv_param) = cache
    stride = conv_param['stride']
    pad = conv_param['pad']

    # Retrieve dimensions
    N, C, H, W = x.shape            # Input dimensions
    F, _, HH, WW = w.shape          # Filter dimensions
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride

    # Initialize gradients
    db = np.zeros((F,))
    dw = np.zeros((F, C, HH, WW))
    dx = np.zeros((N, C, H, W))
    x_padded = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
    dx_padded = np.zeros_like(x_padded)

    # 2. Calculate db (Gradient with respect to bias)
    db = np.sum(dout, axis=(0, 2, 3))  # Sum over N, H', W' dimensions
    
    # 3. Calculate dw (Gradient with respect to weights)
    for point in range(N):  # Loop over each point
        for filter in range(F):  # loop over each filter
            for h_out_index in range(H_out):
                for w_out_index in range(W_out):
                    # Calculate top left corner of the current "slice"
                    h_start = h_out_index * stride
                    w_start = w_out_index * stride
                    h_end = h_start + HH
                    w_end = w_start + WW

                    receptive_field = x_padded[point, :, h_start:h_end, w_start:w_end]

                    # Update dw
                    dw[filter] += receptive_field * dout[point, filter, h_out_index, w_out_index]

                    # 4. Calculate dx (Gradient with respect to input x)
                    dx_padded[point, :, h_start:h_end, w_start:w_end] += w[filter] * dout[point, filter, h_out_index, w_out_index]
    
    # Final Step for dx: Unpad dx_padded to match original x shape
    dx = dx_padded[:, :, pad:pad+H, pad:pad+W]
    # ### END CODE HERE ###
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # ### START CODE HERE ###
    """ Here impleemntation is very similar to convolution forward pass but no padding is applied.
        Also, there are no multiple filters and no weights and biases involved.
        Instead, we first calculate shape and contents of receptive field and 
        take the maximum value from that region as the result. And fill the output accordingly.
    """
    # Extract dimensions and parameters from inputs
    N, C, H, W = x.shape            # Input dimensions: N num layers, C channels, H height, W width
    stride = pool_param['stride']
    pool_h = pool_param['pool_height']
    pool_w = pool_param['pool_width']
    
    # Calculate output dimensions
    H_out = 1 + (H - pool_h) // stride
    W_out = 1 + (W - pool_w) // stride
    
    # Initialize output
    out = np.zeros((N, C, H_out, W_out))

    # Perform the max pool operation
    for point in range(N):  # Loop over each point
        for h_out_index in range(H_out):
            for w_out_index in range(W_out):
                # Calculate top left corner of the current "slice"
                h_start = h_out_index * stride
                w_start = w_out_index * stride
                h_end = h_start + pool_h
                w_end = w_start + pool_w
                
                # Get the receptive field
                receptive_field = x[point, :, h_start:h_end, w_start:w_end]

                # Perform max pooling operation (get the max element in the receptive field
                # ensure that we collapse the H and W dimensions and only keep C dimension for commputing
                # max value)
                out[point, :, h_out_index, w_out_index] = np.max(receptive_field, axis=(1, 2))
    
    # ### END CODE HERE ###
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # ### START CODE HERE ###
    """
        1. Retrieve Information from cache: Unpack x and pool_param from the cache.
        2. Initialize dx: Create a NumPy array of zeros dx with the same shape as the original input x.
        3. Iterate Over Output Grid: Use nested loops that mirror the forward pass (N, C, H_out, W_out).
        4. Identify the Winning Index: For each pooling window:
            a. Extract the receptive field (just like the forward pass).
            b. Find the location (index/mask) of the maximum value within that 2D window using NumPy functions like np.argmax() or by creating a boolean mask.
        5. Route the Gradient: Take the upstream gradient value dout[n, c, h_out, w_out] and place it only into the corresponding "winning" position in your dx array, leaving all other positions in that window as zero.
        6. No dw or db: No not need to calculate or return dw or db, as pooling layers have no weights or biases
    """
    x, pool_param = cache
    N, C, H, W = x.shape            # Input dimensions
    
    stride = pool_param['stride']
    pool_h = pool_param['pool_height']
    pool_w = pool_param['pool_width']

    # Calculate output dimensions
    H_out = 1 + (H - pool_h) // stride
    W_out = 1 + (W - pool_w) // stride
    
    dx = np.zeros_like(x)

    # Perform the max pool backward operation
    for point in range(N):  # Loop over each point
        for channel in range(C):
            for h_out_index in range(H_out):
                for w_out_index in range(W_out):
                    # Calculate top left corner of the current "slice"
                    h_start = h_out_index * stride
                    w_start = w_out_index * stride
                    h_end = h_start + pool_h
                    w_end = w_start + pool_w

                    receptive_field = x[point, channel, h_start:h_end, w_start:w_end]
                    
                    max_flat_idx = np.argmax(receptive_field)
                    
                    # np.divmod() performs two operations simultaneously: integer division and mod (remainder)
                    # It takes two numbers (a and b) and returns a tuple (a // b, a % b)
                    # It translates the 1D "winning index" back into its corresponding 2D relative coordinates
                    h_rel, w_rel = np.divmod(max_flat_idx, pool_w)
                    
                    # Calculate the absolute coordinates in the original input image
                    h_abs = h_start + h_rel
                    w_abs = w_start + w_rel

                    # Route the gradient
                    dx[point, channel, h_abs, w_abs] += dout[point, channel, h_out_index, w_out_index]
    
    # ### END CODE HERE ###
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # ### START CODE HERE ###
    N, C, H, W = x.shape
    x_reshaped = x.transpose(0, 2, 3, 1).reshape(-1, C)  # Reshape to (N*H*W, C)
    out_reshaped, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)
    out = out_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2)  # Reshape back to (N, C, H, W)
    # ### END CODE HERE ###
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # ### START CODE HERE ###
    N, C, H, W = dout.shape
    dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, C)  # Reshape to (N*H*W, C)
    dx_reshaped, dgamma, dbeta = batchnorm_backward(dout_reshaped, cache)
    dx = dx_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2)  # Reshape back to (N, C, H, W)
    # ### END CODE HERE ###
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # ### START CODE HERE ###
    N, C, H, W = x.shape
    x_grouped = x.reshape(N, G, C // G, H, W)  # Reshape to (N, G, C//G, H, W)
    mean = np.mean(x_grouped, axis=(2, 3, 4), keepdims=True)
    var = np.var(x_grouped, axis=(2, 3, 4), keepdims=True)
    x_normalized = (x_grouped - mean) / np.sqrt(var + eps)
    x_normalized = x_normalized.reshape(N, C, H, W)  # Reshape back to (N, C, H, W)
    out = gamma * x_normalized + beta
    cache = (x, x_normalized, mean, var, gamma, beta, G, eps)
    # ### END CODE HERE ###
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # ### START CODE HERE ###
    x, x_normalized, mean, var, gamma, beta, G, eps = cache
    N, C, H, W = x.shape
    
    # Gradient w.r.t. beta
    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
    
    # Gradient w.r.t. gamma
    dgamma = np.sum(dout * x_normalized, axis=(0, 2, 3), keepdims=True)
    
    # Gradient w.r.t. x
    dx_normalized = dout * gamma
    dx_normalized_grouped = dx_normalized.reshape(N, G, C // G, H, W)
    x_grouped = x.reshape(N, G, C // G, H, W)
    mean = mean
    var = var
    group_size = (C // G) * H * W
    dx_grouped = (1. / group_size) * (var + eps) ** (-1. / 2.) * (
        group_size * dx_normalized_grouped
        - np.sum(dx_normalized_grouped, axis=(2, 3, 4), keepdims=True)
        - (x_grouped - mean) * (var + eps) ** (-1.0) * np.sum(dx_normalized_grouped * (x_grouped - mean), axis=(2, 3, 4), keepdims=True)
    )
    dx = dx_grouped.reshape(N, C, H, W)
    # ### END CODE HERE ###
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
