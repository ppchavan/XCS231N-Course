import torch
import numpy as np


def sim(z_i, z_j):
    """Normalized dot product between two vectors.

    Inputs:
    - z_i: 1xD tensor.
    - z_j: 1xD tensor.

    Returns:
    - A scalar value that is the normalized dot product between z_i and z_j.
    """
    norm_dot_product = None
    ##############################################################################
    # TODO: Start of your code.                                                  #
    #                                                                            #
    # HINT: torch.linalg.norm might be helpful.                                  #
    ##############################################################################

    # ### START CODE HERE ###
    # Calculate the dot product of the two vectors (numerator: z_i . z_j)
    dot_product = torch.dot(z_i.flatten(), z_j.flatten())

    # Calculate the L2 norm (magnitude) of each vector (denominator: || z_i || * || z_j ||)
    norm_i = torch.linalg.norm(z_i, ord=2)
    norm_j = torch.linalg.norm(z_j, ord=2)
    
    # Normalize the dot product
    norm_dot_product = dot_product / (norm_i * norm_j)
    # ### END CODE HERE ###

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return norm_dot_product


def simclr_loss_naive(out_left, out_right, tau):
    """Compute the contrastive loss L over a batch (naive loop version).

    Input:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch. The same row in out_left and out_right form a positive pair.
    In other words, (out_left[k], out_right[k]) form a positive pair for all k=0...N-1.
    - tau: scalar value, temperature parameter that determines how fast the exponential increases.

    Returns:
    - A scalar value; the total loss across all positive pairs in the batch. See notebook for definition.
    """
    N = out_left.shape[0]  # total number of training examples

    # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]

    total_loss = 0
    for k in range(N):  # loop through each positive pair (k, k+N)
        z_k, z_k_N = out[k], out[k + N]

        ##############################################################################
        # TODO: Start of your code.                                                  #
        #                                                                            #
        # Hint: Compute l(k, k+N) and l(k+N, k).                                     #
        ##############################################################################
        # ### START CODE HERE ###
        # 1. Calculate Loss for the First Sample in the Pair        
        # 1 a. Calculate Numerator: Computes the similarity between the positive pair (z_k and z_k_N) 
        # and applies the temperature-scaled exponential (\(\exp (\text{sim}(z_{k},z_{kN})/\tau )\))
        numerator_k = torch.exp(sim(z_k, z_k_N) / tau)

        # 1 b. Calculate Denominator: Uses an inner loop to sum the temperature-scaled exponentials 
        # of the similarity between z_k and every other vector z_m in the entire 2N batch 
        # (excluding itself). This sum acts as the normalization term.
        denominator_k = 0
        for m in range(2 * N):
            if m != k:
                denominator_k += torch.exp(sim(z_k, out[m]) / tau)
        
        # 1 c. Calculate Log-Loss: Applies the negative logarithm (\(\text{-log}\)) to 
        # the ratio of the numerator and denominator to get the loss value l_k_k_N 
        # (Normalized Temperature-Scaled Cross Entropy Loss).
        loss_k = -torch.log(numerator_k / denominator_k)

        # 2. Calculate Loss for the Second Sample in the Pair l(k+N, k): 
        # Loss for z_k_N using z_k as the positive example

        # 2 a. Calculate Numerator: Computes the similarity between the positive pair (z_k_N and z_k)
        numerator_k_N = torch.exp(sim(z_k_N, z_k) / tau)

        # 2 b. Calculate Denominator: Sums the temperature-scaled exponentials of the similarity
        denominator_k_N = 0
        for m in range(2 * N):
            if m != k + N:
                denominator_k_N += torch.exp(sim(z_k_N, out[m]) / tau)
        
        # 2 c. Calculate Log-Loss: Applies the negative logarithm to the ratio of the numerator
        loss_k_N = -torch.log(numerator_k_N / denominator_k_N)
        # 3. Accumulate Total Loss: Adds both loss values to the total loss for the batch.
        total_loss += loss_k + loss_k_N      

        # ### END CODE HERE ###
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

    # In the end, we need to divide the total loss by 2N, the number of samples in the batch.
    total_loss = total_loss / (2 * N)
    return total_loss


def sim_positive_pairs(out_left, out_right):
    """Normalized dot product between positive pairs.

    Inputs:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch.
    The same row in out_left and out_right form a positive pair.

    Returns:
    - A Nx1 tensor; each row k is the normalized dot product between out_left[k] and out_right[k].
    """
    pos_pairs = None

    ##############################################################################
    # TODO: Start of your code.                                                  #
    #                                                                            #
    # HINT: torch.linalg.norm might be helpful.                                  #
    ##############################################################################

    # ### START CODE HERE ###
    N = out_left.shape[0]
    pos_pairs = torch.zeros(N, 1, device=out_left.device)
    for k in range(N):
        z_i = out_left[k]
        z_j = out_right[k]
        dot_product = torch.dot(z_i.flatten(), z_j.flatten())
        norm_i = torch.linalg.norm(z_i, ord=2)
        norm_j = torch.linalg.norm(z_j, ord=2)
        norm_dot_product = dot_product / (norm_i * norm_j)
        pos_pairs[k] = norm_dot_product
    # ### END CODE HERE ###

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return pos_pairs


def compute_sim_matrix(out):
    """Compute a 2N x 2N matrix of normalized dot products between all pairs of augmented examples in a batch.

    Inputs:
    - out: 2N x D tensor; each row is the z-vector (output of projection head) of a single augmented example.
    There are a total of 2N augmented examples in the batch.

    Returns:
    - sim_matrix: 2N x 2N tensor; each element i, j in the matrix is the normalized dot product between out[i] and out[j].
    """
    sim_matrix = None

    ##############################################################################
    # TODO: Start of your code.                                                  #
    ##############################################################################

    # ### START CODE HERE ###
    num_samples = out.shape[0]
    sim_matrix = torch.zeros((num_samples, num_samples), device=out.device)
    for i in range(num_samples):
        for j in range(num_samples):
            sim_matrix[i, j] = sim(out[i], out[j])
    # ### END CODE HERE ###

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return sim_matrix


def simclr_loss_vectorized(out_left, out_right, tau, device="cuda"):
    """Compute the contrastive loss L over a batch (vectorized version). No loops are allowed.

    Inputs and output are the same as in simclr_loss_naive.
    """
    N = out_left.shape[0]

    # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]

    # Compute similarity matrix between all pairs of augmented examples in the batch.
    sim_matrix = compute_sim_matrix(out)  # [2*N, 2*N]

    ##############################################################################
    # TODO: Start of your code. Follow the hints.                                #
    ##############################################################################

    # Step 1: Use sim_matrix to compute the denominator value for all augmented samples.
    # Hint: Compute e^{sim / tau} and store into exponential, which should have shape 2N x 2N.
    exponential = None
    exponential = torch.exp(sim_matrix / tau) # [2*N, 2*N]

    # This binary mask zeros out terms where k=i.
    mask = (
        (torch.ones_like(exponential, device=device) - torch.eye(2 * N, device=device))
        .to(device)
        .bool()
    )

    # We apply the binary mask.
    exponential = exponential.masked_select(mask).view(2 * N, -1)  # [2*N, 2*N-1]

    # Hint: Compute the denominator values for all augmented samples. This should be a 2N x 1 vector.
    denom = torch.sum(exponential, dim=1, keepdim=True)  # [2*N, 1]

    # Step 2: Compute similarity between positive pairs.
    # You can do this in two ways:
    # Option 1: Extract the corresponding indices from sim_matrix.
    # Option 2: Use sim_positive_pairs().
    # ### START CODE HERE ###
    sim_pos = None  # [2*N, 1]
    sim_pos = sim_positive_pairs(out_left, out_right)
    sim_pos = torch.cat([sim_pos, sim_pos], dim=0)  # [2*N, 1] 
    # ### END CODE HERE ###

    # Step 3: Compute the numerator value for all augmented samples.
    numerator = None
    # ### START CODE HERE ###
    numerator = torch.exp(sim_pos / tau)  # [2*N, 1]
    # ### END CODE HERE ###

    # Step 4: Now that you have the numerator and denominator for all augmented samples, compute the total loss.
    loss = None
    # ### START CODE HERE ###
    individual_loss = -torch.log(numerator / denom) # [2*N, 1]
    loss = torch.mean(individual_loss)
    # ### END CODE HERE ###

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return loss


def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
