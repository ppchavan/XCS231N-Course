import torch
import torch.nn as nn
from torch.nn import functional as F
import math

"""
This file defines layer types that are commonly used for transformers.
"""


class PositionalEncoding(nn.Module):
    """
    Encodes information about the positions of the tokens in the sequence. In
    this case, the layer has no learnable parameters, since it is a simple
    function of sines and cosines.
    """

    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        Construct the PositionalEncoding layer.

        Inputs:
         - embed_dim: the size of the embed dimension
         - dropout: the dropout value
         - max_len: the maximum possible length of the incoming sequence
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        # Create an array with a "batch dimension" of 1 (which will broadcast
        # across all examples in the batch).
        pe = torch.zeros(1, max_len, embed_dim)
        ############################################################################
        # TODO: Construct the positional encoding array as described in            #
        # Transformer_Captioning.ipynb.  The goal is for each row to alternate     #
        # sine and cosine, and have exponents of 0, 0, 2, 2, 4, 4, etc. up to      #
        # embed_dim. Of course this exact specification is somewhat arbitrary, but #
        # this is what the autograder is expecting. For reference, our solution is #
        # less than 5 lines of code.                                               #
        ############################################################################
        # ### START CODE HERE ###
        # Formula reference:
        # PE(pos, 2i) = sin(pos / (10000^(2i/embed_dim)))
        # PE(pos, 2i+1) = cos(pos / (10000^(2i/embed_dim)))
        
        # Step 1. Example (if max_len is 5): [[0],,,,] (Shape: (5, 1)) create a clumn vector for positions
        position = torch.arange(0, max_len).unsqueeze(1)  # Shape: (max_len, 1)
        
        # Step 2. Create the div_term for the denominator (Shape: (embed_dim/2,))
        # torch.arange(0, embed_dim, 2) creates indices for even positions
        # -math.log(10000.0) / embed_dim is the scaling factor for the exponent which is applied to all even indices
        # (the inside term is mathematically: 2i * (-ln(10000)/d_model) )
        # and we apply torch.exp so it becomes e^(2i * (-ln(10000)/d_model))
        inside_term = torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        div_term = torch.exp(inside_term)  # Shape: (embed_dim/2,)
        
        # Step 3. Apply sine function to even indices and cosine to odd indices
        pe[0, :, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[0, :, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
        # ### END CODE HERE ###
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # Make sure the positional encodings will be saved with the model
        # parameters (mostly for completeness).
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Element-wise add positional embeddings to the input sequence.

        Inputs:
         - x: the sequence fed to the positional encoder model, of shape
              (N, S, D), where N is the batch size, S is the sequence length and
              D is embed dim
        Returns:
         - output: the input sequence + positional encodings, of shape (N, S, D)
        """
        N, S, D = x.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, D))
        ############################################################################
        # TODO: Index into your array of positional encodings, and add the         #
        # appropriate ones to the input sequence. Don't forget to apply dropout    #
        # afterward. This should only take a few lines of code.                    #
        ############################################################################
        # ### START CODE HERE ###
        # Add positional encoding to input x
        output = x + self.pe[:, :S, :]
        output = self.dropout(output)
        # ### END CODE HERE ###
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return output


class MultiHeadAttention(nn.Module):
    """
    A model layer which implements a simplified version of masked attention, as
    introduced by "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).

    Usage:
      attn = MultiHeadAttention(embed_dim, num_heads=2)

      # self-attention
      data = torch.randn(batch_size, sequence_length, embed_dim)
      self_attn_output = attn(query=data, key=data, value=data)

      # attention using two inputs
      other_data = torch.randn(batch_size, sequence_length, embed_dim)
      attn_output = attn(query=data, key=other_data, value=other_data)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Construct a new MultiHeadAttention layer.

        Inputs:
         - embed_dim: Dimension of the token embedding
         - num_heads: Number of attention heads
         - dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        # We will initialize these layers for you, since swapping the ordering
        # would affect the random number generation (and therefore your exact
        # outputs relative to the autograder). Note that the layers use a bias
        # term, but this isn't strictly necessary (and varies by
        # implementation).
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(dropout)

        self.n_head = num_heads
        self.emd_dim = embed_dim
        self.head_dim = self.emd_dim // self.n_head

    def forward(self, query, key, value, attn_mask=None):
        """
        Calculate the masked attention output for the provided data, computing
        all attention heads in parallel.

        In the shape definitions below, N is the batch size, S is the source
        sequence length, T is the target sequence length, and E is the embedding
        dimension.

        Inputs:
        - query: Input data to be used as the query, of shape (N, S, E)
        - key: Input data to be used as the key, of shape (N, T, E)
        - value: Input data to be used as the value, of shape (N, T, E)
        - attn_mask: Array of shape (S, T) where mask[i,j] == 0 indicates token
          i in the source should not influence token j in the target.

        Returns:
        - output: Tensor of shape (N, S, E) giving the weighted combination of
          data in value according to the attention weights calculated using key
          and query.
        """
        N, S, E = query.shape
        N, T, E = value.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, E))
        ############################################################################
        # TODO: Implement multiheaded attention using the equations given in       #
        # Transformer_Captioning.ipynb.                                            #
        # A few hints:                                                             #
        #  1) You'll want to split your shape from (N, T, E) into (N, T, H, E/H),  #
        #     where H is the number of heads.                                      #
        #  2) The function torch.matmul allows you to do a batched matrix multiply.#
        #     For example, you can do (N, H, T, E/H) by (N, H, E/H, T) to yield a  #
        #     shape (N, H, T, T). For more examples, see                           #
        #     https://pytorch.org/docs/stable/generated/torch.matmul.html          #
        #  3) For applying attn_mask, think how the scores should be modified to   #
        #     prevent a value from influencing output. Specifically, the PyTorch   #
        #     function masked_fill may come in handy.                              #
        ############################################################################
        # ### START CODE HERE ###
        # 1. Apply linear projections to get Q, K, V
        # Q, K, V have shapes (N, S, E), (N, T, E), (N, T, E) respectively
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)

        # 2. K: Input data to be used as the key, of shape (N, T, E)
        # where N is the batch size, T is the target sequence length, and E is the embedding dimension. 
        # We need to perform calculations independently for each head in a batched manner. 
        # To do this efficiently with matrix multiplication functions like torch.matmul, 
        # the Heads (H) dimension needs to be moved after the Batch (N) dimension 
        # 
        # After view shape: (Batch Size, Target Sequence Length, Number of Heads, Head Dimension)
        # Or (N, T, H, E/H)
        # then after transpose shape: (N, H, T, E/H)
        Q = Q.view(N, S, self.n_head, self.head_dim).transpose(1, 2)
        K = K.view(N, T, self.n_head, self.head_dim).transpose(1, 2)
        V = V.view(N, T, self.n_head, self.head_dim).transpose(1, 2)
    
        # 3.Calculate attention scores = (Q @ K.T / sqrt(d_k))        
        # After transpose(-2, -1), K shape becomes (N, H, E/H, T)
        # Scores shape: (N, H, S, T)
        # Scaling factor is square root of the dimension of the keys for a single attention head
        # It is calculated as dk = d/h where d is embed_dim and h is num_heads (calculated in init)
        # This scaling factor helps to normalize attention scores and prevent them from growing too large
        scaling_factor = self.head_dim ** 0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scaling_factor
        
        # 4. Apply the attention mask (if provided) to the scores
        if attn_mask is not None:
            # We use float('-inf') to ensure that when softmax is applied,
            # these masked positions receive a weight of 0.
            # Because e raised to (-inifinity) is 0.
            # attn_mask shape: (S, T)
            # We need to unsqueeze to make it broadcastable to scores shape: (N, H, S, T)
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        
        # 5. Apply softmax and dropout. Softmax is applied to last dimension T
        #    which is target sequence length.
        #    Result is a tensor of attn_weights shape: (N, H, S, T)
        #    Dropout is applied to the attention weights to prevent overfitting.
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        # 6. Apply attention weights to V
        # The weights calculated in the previous step are used to 
        # create a weighted average of the Value (V) vectors
        # output shape: (N, H, S, head_dim)
        output = torch.matmul(attn_weights, V)

        # 7. Concatenate heads
        # The individual results from each of the attention heads must be "concatenated"
        # to return the data to its original embedding dimension size.
        # Transpose back to (N, S, H, head_dim) and then reshape to (N, S, E)
        output = output.transpose(1, 2).contiguous().view(N, S, E)

        # 8. Apply final linear projection
        # Linear transform allows the model to mix the information learned across the 
        # different heads in a trainable manner
        # output shape: (N, S, E)
        output = self.proj(output)
        # ### END CODE HERE ###
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return output


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        """
        Simple two-layer feed-forward network with dropout and ReLU activation.

        Inputs:
         - embed_dim: Dimension of input and output embeddings
         - ffn_dim: Hidden dimension in the feedforward network
         - dropout: Dropout probability
        """
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x):
        """
        Forward pass for the feedforward network.

        Inputs:
        - x: Input tensor of shape (N, T, D)

        Returns:
        - out: Output tensor of the same shape as input
        """
        out = torch.empty_like(x)

        out = self.fc1(x)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class TransformerDecoderLayer(nn.Module):
    """
    A single layer of a Transformer decoder, to be used with TransformerDecoder.
    """

    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        """
        Construct a TransformerDecoderLayer instance.

        Inputs:
         - input_dim: Number of expected features in the input.
         - num_heads: Number of attention heads
         - dim_feedforward: Dimension of the feedforward network model.
         - dropout: The dropout value.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(input_dim, dim_feedforward, dropout)

        self.norm_self = nn.LayerNorm(input_dim)
        self.norm_cross = nn.LayerNorm(input_dim)
        self.norm_ffn = nn.LayerNorm(input_dim)

        self.dropout_self = nn.Dropout(dropout)
        self.dropout_cross = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None):
        """
        Pass the inputs (and mask) through the decoder layer.

        Inputs:
        - tgt: the sequence to the decoder layer, of shape (N, T, D)
        - memory: the sequence from the last layer of the encoder, of shape (N, S, D)
        - tgt_mask: the parts of the target sequence to mask, of shape (T, T)

        Returns:
        - out: the Transformer features, of shape (N, T, W)
        """

        # Self-attention block (reference implementation)
        shortcut = tgt
        tgt = self.self_attn(query=tgt, key=tgt, value=tgt, attn_mask=tgt_mask)
        tgt = self.dropout_self(tgt)
        tgt = tgt + shortcut
        tgt = self.norm_self(tgt)

        ############################################################################
        # TODO: Complete the decoder layer by implementing the remaining two       #
        # sublayers: (1) the cross-attention block using the encoder output as     #
        # memory, and (2) the feedforward block. Each block should follow the      #
        # same structure as self-attention implemented just above.                 #
        ############################################################################
        # ### START CODE HERE ###
        # Cross-attention block
        shortcut = tgt
        tgt = self.cross_attn(query=tgt, key=memory, value=memory, attn_mask=None)
        tgt = self.dropout_cross(tgt)
        tgt = tgt + shortcut
        tgt = self.norm_cross(tgt)
        # Feedforward block
        shortcut = tgt
        tgt = self.ffn(tgt)
        tgt = self.dropout_ffn(tgt)
        tgt = tgt + shortcut
        tgt = self.norm_ffn(tgt)
        # ### END CODE HERE ###
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return tgt


class PatchEmbedding(nn.Module):
    """
    A layer that splits an image into patches and projects each patch to an embedding vector.
    Used as the input layer of a Vision Transformer (ViT).

    Inputs:
    - img_size: Integer representing the height/width of input image (assumes square image).
    - patch_size: Integer representing height/width of each patch (square patch).
    - in_channels: Number of input image channels (e.g., 3 for RGB).
    - embed_dim: Dimension of the linear embedding space.
    """

    def __init__(self, img_size, patch_size, in_channels=3, embed_dim=128):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        #print(img_size, patch_size)
        assert (
            img_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."

        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * in_channels

        # Linear projection of flattened patches to the embedding dimension
        self.proj = nn.Linear(self.patch_dim, embed_dim)

    def forward(self, x):
        """
        Forward pass for patch embedding.

        Inputs:
        - x: Input image tensor of shape (N, C, H, W)

        Returns:
        - out: Patch embeddings with positional encodings of shape (N, num_patches, embed_dim)
        """
        N, C, H, W = x.shape
        assert (
            H == self.img_size and W == self.img_size
        ), f"Expected image size ({self.img_size}, {self.img_size}), but got ({H}, {W})"
        out = torch.zeros(N, self.embed_dim)

        ############################################################################
        # TODO: Divide the image into non-overlapping patches of shape             #
        # (patch_size x patch_size x C), and rearrange them into a tensor of       #
        # shape (N, num_patches, patch_dim). Do not use a for-loop.                #
        # Instead, you may find torch.reshape and torch.permute helpful for this   #
        # step. Once the patches are flattened, embed them into latent vectors     #
        # using the projection layer.                                              #
        ############################################################################
        # ### START CODE HERE ###
        # Step 1: Rearrange image into patches
        #print(f"Input shape: {x.shape}")
        """ 
        Here x is of shape (N, C, H, W)
            N is batch size (num images in the batch), 
            C is number of channels (for RGB this is 3), 
            H is height of image, 
            W is width of image, here H=W=img_size        
        """
        height_dim_idx = 2  # Height dimension index
        width_dim_idx = 3   # Width dimension index
        # If we set step=self.patch_size, we get non-overlapping patches for Vision transformer
        patches = x.unfold(height_dim_idx, size=self.patch_size, step=self.patch_size)

        # After first unfold, shape becomes (N, C, num_patches_h, W, patch_size)
        patches = patches.unfold(width_dim_idx, self.patch_size, self.patch_size)
        # After second unfold, shape becomes (N, C, num_patches_h, num_patches_w, patch_size, patch_size)
        # If image size = 16 and patch size = 8 8
        # Input x shape: torch.Size([2, 3, 16, 16])
        # Unfolded shape 1: torch.Size([2, 3, 2, 16, 8])
        # Unfolded shape 2: torch.Size([2, 3, 2, 2, 8, 8])
        
        # Step 2: Rearrange dimensions to get patches in the right order
        # Shape after permute: (N, num_patches_h, num_patches_w, C, patch_size, patch_size)
        # We did this to safely apply a .reshape() operation to merge the 
        # last three dimensions into a single patch_dim
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        
        # Step 3: Reshape to get (N, num_patches, patch_dim)
        patches = patches.reshape(N, self.num_patches, self.patch_dim)

        #print(f"Num images: {N}\nNumber of patches: {self.num_patches}\nPatch dimension: {self.patch_dim}")
        #print(f"Patches shape before projection: {patches.shape}")
        
        # Step 4: Apply linear projection to get embeddings
        out = self.proj(patches)
        #print(f"Output shape after projection: {out.shape}")

        # ### END CODE HERE ###
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return out


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of a Transformer encoder, to be used with TransformerEncoder.
    """

    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        """
        Construct a TransformerEncoderLayer instance.

        Inputs:
         - input_dim: Number of expected features in the input.
         - num_heads: Number of attention heads.
         - dim_feedforward: Dimension of the feedforward network model.
         - dropout: The dropout value.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(input_dim, dim_feedforward, dropout)

        self.norm_self = nn.LayerNorm(input_dim)
        self.norm_ffn = nn.LayerNorm(input_dim)

        self.dropout_self = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        """
        Pass the inputs (and mask) through the encoder layer.

        Inputs:
        - src: the sequence to the encoder layer, of shape (N, S, D)
        - src_mask: the parts of the source sequence to mask, of shape (S, S)

        Returns:
        - out: the Transformer features, of shape (N, S, D)
        """
        ############################################################################
        # TODO: Implement the encoder layer by applying self-attention followed    #
        # by a feedforward block. This code will be very similar to decoder layer. #
        ############################################################################
        # ### START CODE HERE ###
        # Self-attention block
        shortcut = src
        src = self.self_attn(query=src, key=src, value=src, attn_mask=src_mask)
        src = self.dropout_self(src)
        src = src + shortcut
        src = self.norm_self(src)
        # Feedforward block
        shortcut = src
        src = self.ffn(src)
        src = self.dropout_ffn(src)
        src = src + shortcut
        src = self.norm_ffn(src)
        # ### END CODE HERE ###
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return src
