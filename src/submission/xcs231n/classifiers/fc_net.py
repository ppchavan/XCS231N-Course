from builtins import range
from builtins import object
import numpy as np
import traceback

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.hidden_dims = hidden_dims
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # ### START CODE HERE ###
        layer_input_dim = input_dim

        # Iterate through the hidden layers first
        for i, h_dim in enumerate(hidden_dims):
            self.params['W' + str(i+1)] = np.random.normal(scale=weight_scale, size=(layer_input_dim, h_dim))
            self.params['b' + str(i+1)] = np.zeros(h_dim)
            
            if self.normalization == 'batchnorm' or self.normalization == 'layernorm':
                self.params['gamma' + str(i+1)] = np.ones(h_dim)
                self.params['beta' + str(i+1)] = np.zeros(h_dim)
                
            layer_input_dim = h_dim

        # Handle the final output layer separately
        # i is now the correct index for the final layer (num_layers)
        i = self.num_layers 
        self.params['W' + str(i)] = np.random.normal(scale=weight_scale, size=(layer_input_dim, num_classes))
        self.params['b' + str(i)] = np.zeros(num_classes)
        # ### END CODE HERE ###
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # ### START CODE HERE ###
        """
            The standard order of operations in the forward pass is 
            - Fully Connected (affine), 
            - Batch Normalization, 
            - ReLU activation, 
            - Dropout
            - Repeat for all hidden layer
            - Final Fully Connected layer (affine) to get scores
        """
        try:
            ar_cache = {} # Affine BatchNorm ReLU Cache
            dp_cache = {} # DropOut Cache
            layer_input = X
            for i in range(1, self.num_layers):
                # 1. Affine layer
                label = f'W{i}'
                print("Processing forward pass for label:", label)
                affine_out, affine_cache = affine_forward(layer_input, self.params[f'W{i}'], self.params[f'b{i}'])

                #2. Normalization layer
                if self.normalization:
                    if self.normalization == 'batchnorm':
                        norm_out, norm_cache = batchnorm_forward(affine_out,
                                        self.params[f'gamma{i}'],
                                        self.params[f'beta{i}'],
                                        self.bn_params[i-1])    
                        
                    elif self.normalization == 'layernorm':
                        norm_out, norm_cache = layernorm_forward(affine_out,
                                        self.params[f'gamma{i}'],
                                        self.params[f'beta{i}'],
                                        self.bn_params[i-1])
                    
                    # The output of batchnorm is the input to ReLU
                    relu_out, relu_cache = relu_forward(norm_out)

                    # Combine all caches for backward pass
                    layer_input = relu_out
                    ar_cache[i] = (affine_cache, norm_cache, relu_cache) # Store all three caches
                else:
                    # No normalization: Affine -> ReLU
                    relu_out, relu_cache = relu_forward(affine_out)
                    layer_input = relu_out
                    ar_cache[i] = (affine_cache, relu_cache) # Store affine and relu caches
                
                if self.use_dropout:
                    layer_input, dp_cache[i] = dropout_forward(layer_input, self.dropout_param)
                    
            #Last layer (no ReLU)
            print("Processing final layer")
            label = f'W{self.num_layers}'
            print("Processing forward pass for label:", label)
            scores, ar_cache[self.num_layers] = affine_forward(layer_input, self.params[f'W{self.num_layers}'], self.params[f'b{self.num_layers}'])
            print("Completed forward pass.")
            # ### END CODE HERE ###
            ############################################################################
            #                             END OF YOUR CODE                             #
            ############################################################################

            # If test mode return early.
            if mode == "test":
                return scores

            loss, grads = 0.0, {}
            ############################################################################
            # TODO: Implement the backward pass for the fully connected net. Store the #
            # loss in the loss variable and gradients in the grads dictionary. Compute #
            # data loss using softmax, and make sure that grads[k] holds the gradients #
            # for self.params[k]. Don't forget to add L2 regularization!               #
            #                                                                          #
            # When using batch/layer normalization, you don't need to regularize the   #
            # scale and shift parameters.                                              #
            #                                                                          #
            # NOTE: To ensure that your implementation matches ours and you pass the   #
            # automated tests, make sure that your L2 regularization includes a factor #
            # of 0.5 to simplify the expression for the gradient.                      #
            ############################################################################
            # ### START CODE HERE ###
            #Compute softmax loss
            loss, dh_out = softmax_loss(scores, y)
            print(f"Computed softmax loss")

            # Add L2 regularization loss for ALL weights
            for i in range(1, self.num_layers + 1):
                loss += 0.5 * self.reg * np.sum(self.params[f'W{i}'] * self.params[f'W{i}'])
            
            # Backward propogate from last layer to input layer
            dx,dw,db = affine_backward(dh_out, ar_cache[self.num_layers])
            print(f"Completed backward pass for layer {i}")

            grads[f'W{self.num_layers}'] = dw + self.reg * self.params[f'W{self.num_layers}']
            grads[f'b{self.num_layers}'] = db
            print(f"Calcualted grads for {i}")
            
            #This is upstream gradient
            dh_out = dx

            # Gradient flow: Should be in this order
            # Dropout (if used) -> ReLU -> BatchNorm (if used) -> Affine
            for i in range(self.num_layers - 1, 0, -1):
                # Gradient flow: Dropout (if used) -> ReLU -> BatchNorm (if used) -> Affine

                # A. Dropout backward (if used)
                if self.use_dropout:
                    dh_out = dropout_backward(dh_out, dp_cache[i])

                # B. Normalization/ReLU backward
                if self.normalization:
                    # Unpack the cache stored as (affine_cache, norm_cache, relu_cache) in forward pass
                    affine_cache, norm_cache, relu_cache = ar_cache[i] 
                    
                    # Backprop through ReLU
                    dout_norm = relu_backward(dh_out, relu_cache)

                    if self.normalization == 'batchnorm':                 
                        # Backprop through BatchNorm (returns dx, dgamma, dbeta)
                        dout_affine, dgamma, dbeta = batchnorm_backward(dout_norm, norm_cache)
                    elif self.normalization == 'layernorm':
                        # Backprop through LayerNorm (returns dx, dgamma, dbeta)
                        dout_affine, dgamma, dbeta = layernorm_backward(dout_norm, norm_cache)
                        
                    # Backprop through Affine (returns dx, dw, db)
                    dx, dw, db = affine_backward(dout_affine, affine_cache)
                            
                    # Store gradients for gamma/beta (no regularization needed)
                    grads[f'gamma{i}'] = dgamma
                    grads[f'beta{i}'] = dbeta
                else:
                    # No normalization: Unpack the cache stored as (affine_cache, relu_cache) in forward pass                    
                    affine_cache, relu_cache = ar_cache[i]
                    
                    # Backprop through ReLU
                    dout_affine = relu_backward(dh_out, relu_cache)

                    # Backprop through Affine
                    dx, dw, db = affine_backward(dout_affine, affine_cache)

                # C. Store gradients for W and b (with L2 regularization)
                grads[f'W{i}'] = dw + self.reg * self.params[f'W{i}']
                grads[f'b{i}'] = db
                
                # D. Update upstream gradient for the next iteration
                dh_out = dx
                print(f"Completed backward pass for layer {i}")
            print("Completed backward pass. All gradients computed.")
            # ### END CODE HERE ###
            ############################################################################
            #                             END OF YOUR CODE                             #
            ############################################################################
        except Exception as e:
            print("An error occurred during loss computation:")
            print(str(e))
            traceback.print_exc()            
            raise e
        
        return loss, grads
