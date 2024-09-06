
import torch
from torch import nn

from einops import rearrange, repeat
from timm.models.layers import trunc_normal_  # Pytorch-based image model

from util import windowBoldSignal


"""
function:
    max_neg_value(tensor) find how precise the negative values can be in the PyTorch tensor object

inputs:
    tensor - pytorch tensor object

outputs:
    returns the negative of the maximum representable finite floating-point number for a given data type in PyTorch
"""

def max_neg_value(tensor):

    # torch.finfo.max represents the maximum representable finite floating-point number for a given data type in PyTorch
    return -torch.finfo(tensor.dtype).max

class PreNorm(nn.Module):

    """
    function:
        __init__(self, dim, fn) initializes the build-in function .norm() and the attribute fn. The LayerNorm(dim)
        function normalizes the value of a given input object (n-dimensional). It performs a per-element normalization
        and outputs an object of same dimension as the input object with normalized elements. It is important to note
        that the LayerNorm() function has learnable weights and bias terms to better normalize the elements

    inputs:
        dim - dimensions of the layer to normalize
        fn - blank object to assign function to?

    outputs:
        embeds the LayerNorm function and fn attribute to the object
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    """
    function:
        forward(self, x, **kwargs) creates a function that utilizes the norm() built-in (defined directly above) and 
        has it return a layer normalized object when called. Call this function once class is applied to a Pytorch 
        tensor object
        
    inputs:
        x - dimensions of the layer to normalize
        **kwargs - additional options to be defined

    outputs:
        performs a layer normalization using the norm() function created directly above
    """

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):

    """
    function:
        __init__(self, dim, mult = 1, dropout = 0.,) initializes a fully-connected, forward feed neural network using
        a two linear neural net layers (affine transform) with learnable weight matrix and bias terms. The first layer
        uses the GELU activation function, and the second uses the RELU activation function. Additionally, the user can
        specify the dropout probability of values (zeroing of values) when the tensor is passed from the first layer to
        the second

        The zeroing of values of a tensor (dropping out) between neural network layers has shown to be an
        effective technique for regularization and preventing the co-adaptation of neurons

    inputs:
        dim - dimension of output from the forward feed network
        mult - additional parameter to determine the output of the forward feed network
        dropout - probability to randomly 0 elements in a tensor passed between neural network layers

    outputs:
        embeds a two linear, fully-connected layer neural network into the python object
    """

    def __init__(self, dim, mult = 1, dropout = 0.,):
        super().__init__()
        inner_dim = int(dim * mult) # output of the first linear layer
        dim_out = dim  # output dimension of second linear layer
        activation = nn.GELU()  # applies the Gaussian Error Linear Units activation function

        # nn.Sequential builds a sequence of neural network layers
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),  # learnable affine transformation from dimension "dim" to dimension "inner_dim"
            activation  # GELU activation function used
        )

        self.net = nn.Sequential(
            project_in,  # gets new tensor from 1st learnable affine transformation layer
            nn.Dropout(dropout),  # randomly zeros out elements of new tensor with probability "dropout"
            nn.Linear(inner_dim, dim_out),  # linear layer from dim. "dim" to dim. "inner_dim" w/ RELU act. function
        )

    """
    function:
        forward(self, x) executes and returns the result from the two linear, fully-connected layer neural network 
        created by the __init()__ function directly above

    inputs:
        x - arguments to pass to the net() built-in function defined directly above

    outputs:
        returns the result from the two linear, fully-connected layer neural network 
        created by the __init()__ function directly above
    """

    def forward(self, x):
        return self.net(x)


class WindowAttention(nn.Module):

    """
    function:
        __init__() loads parameters and attributes into the transformer model. Determines if bias matrix will be applied
        to the attention matrix and context matrix (attention*value matrix result), respectively. Additionally,
        determines if a dropout procedure (zeroing of elements in outputted matrices between steps) will be employed.
        Parameter() creates a subclass of tensors that are added to the list of parameters when used with Module() from
        the Pytorch package

    inputs:
        dim - output dimensions of FW-MSA content (projection) matrix resulting from combining the individual content
              matrices resulting from each attention head in the FW-MSA mechanism
        windowSize - number of sequential BOLD tokens in each window (excludes fringe tokens between adjacent windows)
        receptiveSize - number of sequential BOLD tokens in receptive field (window tokens and fringe tokens)
        numHeads - number of attention heads in the FW-MSA (fused-window multi-attention head) mechanism
        headDim - number of output dimensions for each individual attention head in the FW-MSA mechanism
        attentionBias - True/False argument to add a learnable bias matrix to the attention matrix
        qkvBias - True/False argument to add a learnable bias matrix to the content matrix of each MSA content matrix
        attnDrop - probability that the attention matrix's elements are zeroed out
        projDrop - probability that the content (projection) matrix's from FW-MSA mechanism elements are zeroed out

    outputs:
        creates a set of parameter objects/attributes and functions to be used in the BolT framework (module)
    """

    def __init__(self, dim, windowSize, receptiveSize, numHeads, headDim=20, attentionBias=True, qkvBias=True, attnDrop=0., projDrop=0.):

        super().__init__()
        self.dim = dim
        self.windowSize = windowSize  # N
        self.receptiveSize = receptiveSize # M
        self.numHeads = numHeads
        head_dim = headDim
        self.scale = head_dim ** -0.5

        self.attentionBias = attentionBias

        # define a parameter table of relative position bias
        
        maxDisparity = windowSize - 1 + (receptiveSize - windowSize)//2  # window size plus length of fringe window


        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2*maxDisparity+1, numHeads))  # maxDisparity, nH

        # creates zero matrices of varying dimensions to

        self.cls_bias_sequence_up = nn.Parameter(torch.zeros((1, numHeads, 1, receptiveSize)))
        self.cls_bias_sequence_down = nn.Parameter(torch.zeros(1, numHeads, windowSize, 1))
        self.cls_bias_self = nn.Parameter(torch.zeros((1, numHeads, 1, 1)))

        # get pair-wise relative position index for each token inside the window

        # 1-d tensor of indices for each window token. Values in set [0, N); denote as N
        coords_x = torch.arange(self.windowSize)

        # gives index value for all tokens including fringe tokens shifted by half the number of fringe tokens (L)
        # index values in set [-L, N); denote as M

        coords_x_ = torch.arange(self.receptiveSize) - (self.receptiveSize - self.windowSize)//2
        relative_coords = coords_x[:, None] - coords_x_[None, :]  # combine N, M into a 2-tensor
        relative_coords[:, :] += maxDisparity  # shift to start from 0
        relative_position_index = relative_coords  # (N, M)
        self.register_buffer("relative_position_index", relative_position_index)

        # initialize learnable affine transform for query, value, and key matrices
        # base tokens (window tokens only) fed into "q" and receptive field tokens (window and fringe) fed into "kv"

        self.q = nn.Linear(dim, head_dim * numHeads, bias=qkvBias)
        self.kv = nn.Linear(dim, 2 * head_dim * numHeads, bias=qkvBias)

        # create linear layer for content (projection matrix) and set attention and content element dropout rates
        self.attnDrop = nn.Dropout(attnDrop)
        self.proj = nn.Linear(head_dim * numHeads, dim)

        self.projDrop = nn.Dropout(projDrop)

        # prep the biases
        trunc_normal_(self.relative_position_bias_table, std=.02)
        trunc_normal_(self.cls_bias_sequence_up, std=.02)
        trunc_normal_(self.cls_bias_sequence_down, std=.02)
        trunc_normal_(self.cls_bias_self, std=.02)

        # set dim of softmax function to -1. All values along this dim will sum to 1
        self.softmax = nn.Softmax(dim=-1)

        # for token painting
        self.attentionMaps = None # shape = (#windows * nH, 1+windowSize, 1+receptiveSize)
        self.attentionGradients = None # shape = (#windows * nH, 1+windowSize, 1+receptiveSize)
        self.nW = None

    """
    function:
        records attention mappings from the BolT model
        
    inputs:
        attentionMaps - attention maps outputted from the FW-MSA mechanism
        
    outputs:
        attention mappings from the BolT model
    """

    def save_attention_maps(self, attentionMaps):
        self.attentionMaps = attentionMaps

    """
    function:
        save_attention_gradients(self, grads) records gradient mappings that show classification loss function 
        improvement activation mapping

    inputs:
        grads - gradient maps associated with a given class outputted from a given activation map

    outputs:
        gradient mappings from the BolT model
    """

    def save_attention_gradients(self, grads):
        self.attentionGradients = grads

    """
    function:
        averageJuiceAcrossHeads(self, cam, grad) produces a grad-CAM mappings that show the average feature 
        importance of each attention map produced from each transformer block of from the BolT model. The average 
        feature importance will be assigned to the individual values of the initially inputted data. This function 
        can be used at each block of the BolT model, so the user can identify the most important areas for each 
        classification class

    inputs:
        cam - activation mapping for a singular class based on the activation weights of an inputted attention map
        grads - gradient maps outputted from the FW-MSA mechanism

    outputs:
        outputs a grad-CAM mappings that show the feature importance from the BolT model
    """

    def averageJuiceAcrossHeads(self, cam, grad):

        """
            Hacked from the original paper git repo ref: https://github.com/hila-chefer/Transformer-MM-Explainability
            cam : (numberOfHeads, n, m)
            grad : (numberOfHeads, n, m)
        """

        #cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
        #grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
        cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=0)
        return cam

    """
    function:
        getJuiceFlow(self, shiftSize) combines each time windows grad-CAM mappings to achieve a global grad-CAM 
        mapping. This global mapping shows the average feature importance of the initial inputted dataset for a 
        given subject when classifying a particular class

    inputs:
        shiftSize - how many time units to skip when starting a subsequent time window

    outputs:
        globalJuiceMatrix - matrix that shows average feature importance when classifying on a particular class
    """

    def getJuiceFlow(self, shiftSize):

        # infer the dynamic length (nW is # of windows)
        dynamicLength = self.windowSize + (self.nW - 1) * shiftSize

        # produces attention maps and their gradients for all time windows
        targetAttentionMaps = self.attentionMaps  # (nW, h, n, m)
        targetAttentionGradients = self.attentionGradients  # self.attentionGradients # (nW h n m)

        # initialize and save global grad-CAM matrix and normalizer matrix to the chosen computer hardware (device)
        globalJuiceMatrix = torch.zeros((self.nW + dynamicLength, self.nW + dynamicLength)).to(targetAttentionMaps.device)
        normalizerMatrix = torch.zeros((self.nW + dynamicLength, self.nW + dynamicLength)).to(targetAttentionMaps.device)

        # aggregate (by averaging) the juice from all the windows
        for i in range(self.nW):

            # average the juices across heads
            window_averageJuice = self.averageJuiceAcrossHeads(targetAttentionMaps[i], targetAttentionGradients[i]) # of shape (1+windowSize, 1+receptiveSize)
            
            # now broadcast the juice to the global juice matrix.

            # set boundaries for overflowing focal attentions
            L = (self.receptiveSize-self.windowSize)//2

            overflow_left = abs(min(i*shiftSize - L, 0))
            overflow_right = max(i*shiftSize + self.windowSize + L - dynamicLength, 0)

            leftMarker_global = i*shiftSize - L + overflow_left
            rightMarker_global = i*shiftSize + self.windowSize + L - overflow_right
            
            leftMarker_window = overflow_left
            rightMarker_window = self.receptiveSize - overflow_right

            # first the cls itself
            globalJuiceMatrix[i, i] += window_averageJuice[0,0]
            normalizerMatrix[i, i] += 1

            # cls to bold tokens
            globalJuiceMatrix[i, self.nW + leftMarker_global : self.nW + rightMarker_global] += window_averageJuice[0, 1+leftMarker_window:1+rightMarker_window]
            normalizerMatrix[i, self.nW + leftMarker_global : self.nW + rightMarker_global] += torch.ones_like(window_averageJuice[0, 1+leftMarker_window:1+rightMarker_window])

            # bold tokens to cls
            globalJuiceMatrix[self.nW + i*shiftSize : self.nW + i*shiftSize + self.windowSize, i] += window_averageJuice[1:, 0]
            normalizerMatrix[self.nW + i*shiftSize : self.nW + i*shiftSize + self.windowSize, i] += torch.ones_like(window_averageJuice[1:, 0])

            # bold tokens to bold tokens
            globalJuiceMatrix[self.nW + i*shiftSize : self.nW + i*shiftSize + self.windowSize, self.nW + leftMarker_global : self.nW + rightMarker_global] += window_averageJuice[1:, 1+leftMarker_window:1+rightMarker_window]
            normalizerMatrix[self.nW + i*shiftSize : self.nW + i*shiftSize + self.windowSize, self.nW + leftMarker_global : self.nW + rightMarker_global] += torch.ones_like(window_averageJuice[1:, 1+leftMarker_window:1+rightMarker_window])

        # to prevent divide by zero for those non-existent attention connections
        normalizerMatrix[normalizerMatrix == 0] = 1

        globalJuiceMatrix = globalJuiceMatrix / normalizerMatrix

        return globalJuiceMatrix

    """
    function: 
        forward(self, x, x_, mask, nW, analysis=False)
        
    inputs:
        x - base BOLD tokens with shape of (B*num_windows, 1+windowSize, C), the first one is cls token and C is the
            embedding dimension (?)
        x_ - receptive BOLD tokens with shape of (B*num_windows, 1+receptiveSize, C), again the first one is cls 
             token
        mask - (mask_left, mask_right) with shape (maskCount, 1+windowSize, 1+receptiveSize). Masks tokens when 
               computing the a pairwise comparison 
        nW - number of windows
        analysis - Boolean, it is set True only when you want to analyze the model, not important otherwise 

    outputs:
        x - attended BOLD tokens from the base of the window, shape = (B*num_windows, 1+windowSize, C), the 
        first one is cls token
    """

    def forward(self, x, x_, mask, nW, analysis=False):

        B_, N, C = x.shape
        _, M, _ = x_.shape
        N = N-1
        M = M-1

        B = B_ // nW 

        mask_left, mask_right = mask

        # linear mapping
        # N = window size, M = receptive field size (window plus fringe), C = embedding dim?
        q = self.q(x) # (batchSize * #windows, 1+N, C)
        k, v = self.kv(x_).chunk(2, dim=-1) # (batchSize * #windows, 1+M, C)

        # head seperation (reorders tensors)
        # b = batch size (# of subjects?), n = window size, h = number of heads, d = embedding dim?
        # (h d) -> h represents an axis as a combination of new axes

        # get 4-d decomposed tensor object; isolates query matrix for each head
        q = rearrange(q, "b n (h d) -> b h n d", h=self.numHeads)

        # get 4-d decomposed tensor object; isolates key matrix for each head
        k = rearrange(k, "b m (h d) -> b h m d", h=self.numHeads)

        # get 4-d decomposed tensor object; isolates value matrix for each head
        v = rearrange(v, "b m (h d) -> b h m d", h=self.numHeads)

        # create 4-d attention tensor matrices that have attention matrix for each attention head and subject
        attn = torch.matmul(q , k.transpose(-1, -2)) * self.scale # (batchSize*#windows, h, n, m)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, M, -1)  # N, M, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, N, M
       
        # add bias matrices to the CLS tensor matrices and attention matrices
        if(self.attentionBias):
            attn[:, :, 1:, 1:] = attn[:, :, 1:, 1:] + relative_position_bias.unsqueeze(0)
            attn[:, :, :1, :1] = attn[:, :, :1, :1] + self.cls_bias_self
            attn[:, :, :1, 1:] = attn[:, :, :1, 1:] + self.cls_bias_sequence_up
            attn[:, :, 1:, :1] = attn[:, :, 1:, :1] + self.cls_bias_sequence_down
        
        # mask the not matching queries and tokens here
        maskCount = mask_left.shape[0]
        # repeat masks for batch and heads (creates masks for all batches and attention heads of those batches)
        mask_left = repeat(mask_left, "nM nn mm -> b nM h nn mm", b=B, h=self.numHeads)
        mask_right = repeat(mask_right, "nM nn mm -> b nM h nn mm", b=B, h=self.numHeads)

        mask_value = max_neg_value(attn) 


        attn = rearrange(attn, "(b nW) h n m -> b nW h n m", nW = nW)        
        
        # make sure masks do not overflow
        maskCount = min(maskCount, attn.shape[1])
        mask_left = mask_left[:, :maskCount]
        mask_right = mask_right[:, -maskCount:]

        # apply masks to attention matrices (before value matrix is applied) from all attention heads for all batches
        attn[:, :maskCount].masked_fill_(mask_left == 1, mask_value)
        attn[:, -maskCount:].masked_fill_(mask_right == 1, mask_value)
        attn = rearrange(attn, "b nW h n m -> (b nW) h n m")

        # apply softmax to the set of all masked attention matrices
        attn = self.softmax(attn) # (b, h, n, m)

        if(analysis):
            self.save_attention_maps(attn.detach())  # save attention
            handle = attn.register_hook(self.save_attention_gradients)  # save it's gradient
            self.nW = nW
            self.handle = handle

        # perform dropout element zeroing using probability set prior to this step
        attn = self.attnDrop(attn)

        # compute the context (projection) matrices for each attention head for all batches
        x = torch.matmul(attn, v)  # of shape (b_, h, n, d)

        # concatenate 4-d tensor values along height (vertical axis); outputs 3-d tensor with dims: (b, n, (h * d))
        x = rearrange(x, 'b h n d -> b n (h d)')

        # apply learnable affine transformation on the collection of context matrices to combine into singular matrix
        x = self.proj(x)
        x = self.projDrop(x)
        
        return x



class FusedWindowTransformer(nn.Module):

    """
    function:
        __init__() embeds objects with the functions WindowAttention and FeedForward (see above for both) and the
        parameters associated with those functions. Additionally, __init__() embeds layer normalization functions
        along dimension associated with the argument dim()

    inputs:
        dim - dimensions of the layer to normalize
        windowSize - number of sequential BOLD tokens in each window (excludes fringe tokens between adjacent windows)
        shiftSize - how many time units to skip when starting a subsequent time window
        receptiveSize - number of sequential BOLD tokens in receptive field (window tokens and fringe tokens)
        numHeads - number of attention heads in the FW-MSA (fused-window multi-attention head) mechanism
        headDim - number of output dimensions for each individual attention head in the FW-MSA mechanism
        mlpRatio - additional parameter to determine the output of the forward feed network
        attentionBias - True/False argument to add a learnable bias matrix to the attention matrix
        drop - probability that the content (projection) matrix's from FW-MSA mechanism elements are zeroed out
        attnDrop - probability that the attention matrix's elements are zeroed out

    outputs:
        embeds the functions WindowAttention and FeedForward (see above for both) and the parameters associated with
        those functions. Additionally, __init__() embeds layer normalization functions along dimension associated
        with the argument dim()
    """

    def __init__(self, dim, windowSize, shiftSize, receptiveSize, numHeads, headDim, mlpRatio, attentionBias, drop, attnDrop):
        
        super().__init__()


        self.attention = WindowAttention(dim=dim, windowSize=windowSize, receptiveSize=receptiveSize, numHeads=numHeads, headDim=headDim, attentionBias=attentionBias, attnDrop=attnDrop, projDrop=drop)
        
        self.mlp = FeedForward(dim=dim, mult=mlpRatio, dropout=drop)

        self.attn_norm = nn.LayerNorm(dim)
        self.mlp_norm = nn.LayerNorm(dim)

        self.shiftSize = shiftSize

    """
    function:
        getJuiceFlow(self) runs the getJuiceFlow function on the object that is it embedded in. This will produce a
        matrix that shows the average feature importance for a given context mapping (projection matrix) when 
        classifying on a particular class

    inputs:
        
    outputs:
        returns a matrix that shows average feature importance when classifying on a particular class

    """

    def getJuiceFlow(self):  
        return self.attention.getJuiceFlow(self.shiftSize)

    """
    function:
        forward(self, x, cls, windowX, windowX_, mask, nW, analysis=False) performs a single pass through a BolT 
        transformer block. After the new context matrices for each CLS and BOLD token have been computed from each 
        FW-MSA head, the BOLD tokens whose indices share time windows are fused together. The new tokens are aggregated
        with the input tokens via a residual connection and subsequently pass through and MLP layer
        
    inputs: 
        x - BOLD input tokens to pass through BolT transformer block. Tensor with dimensions (B, T, C)
        cls - CLS input tokens to pass through BolT transformer block. Tensor with dimensions (B, nW, C)
        windowX - gives the dimensions for window size. This input has dimension (B, 1+windowSize, C)
        windowX_ - gives the dimensions for receptive field size (window size + 2*L where L is the fringe token length.
                   This input has the dimension (B, 1+windowReceptiveSize, C)
        mask - provides which values to mask when computing the context matrix. input is of dimension 
               (B, 1+windowSize, 1+windowReceptiveSize)
        nW - number of windows
        analysis : Boolean, it is set True only when you want to analyze the model, otherwise not important 

    outputs:
        xTrans - (B, T, C)
        clsTrans - (B, nW, C)

    """
    def forward(self, x, cls, windowX, windowX_, mask, nW, analysis=False):

        # WINDOW ATTENTION - computes all context matrices from each attention head
        windowXTrans = self.attention(self.attn_norm(windowX), self.attn_norm(windowX_), mask, nW, analysis=analysis) # (B*nW, 1+windowSize, C)
        clsTrans = windowXTrans[:,:1] # extract the CLS token context matrix (B*nW, 1, C)
        xTrans = windowXTrans[:,1:] # extract the BOLD token context matrices (B*nW, windowSize, C)
        
        clsTrans = rearrange(clsTrans, "(b nW) l c -> b (nW l) c", nW=nW)  # each CLS token corresponds to time window
        xTrans = rearrange(xTrans, "(b nW) l c -> b nW l c", nW=nW)

        # FUSION - fuses the BOLD tokens that share different time windows together
        xTrans = self.gatherWindows(xTrans, x.shape[1], self.shiftSize)
        
        # residual connections
        clsTrans = clsTrans + cls
        xTrans = xTrans + x

        # MLP layers
        xTrans = xTrans + self.mlp(self.mlp_norm(xTrans))
        clsTrans = clsTrans + self.mlp(self.mlp_norm(clsTrans))

        return xTrans, clsTrans

    """
    function:
        gatherWindows(self, windowedX, dynamicLength, shiftSize) 
        
        NOT SURE WHAT THIS IS DOING OR WHAT IT IS OUTPUTTING

    inputs: 
        windowedX - tensor of input BOLD tokens within their respective windows for a given batch of subjects. 
                    Embeddings are of dimension C. There are a nW windows for each subject's set of BOLD tokens
                    This input has the following dimension (batchSize, nW, windowLength, C)
        dynamicLength - The window size?
        shiftSize - how many time units to skip when starting a subsequent time window

    outputs:
        destination - (batchSize, dynamicLength, C)

    """
    def gatherWindows(self, windowedX, dynamicLength, shiftSize):

        batchSize = windowedX.shape[0]
        windowLength = windowedX.shape[2]
        nW = windowedX.shape[1]
        C = windowedX.shape[-1]
        
        device = windowedX.device


        destination = torch.zeros((batchSize, dynamicLength,  C)).to(device)
        scalerDestination = torch.zeros((batchSize, dynamicLength, C)).to(device)

        # creates a 2-d tensor where each row is the set of the index values of the ith window
        indexes = torch.tensor([[j+(i*shiftSize) for j in range(windowLength)] for i in range(nW)]).to(device)

        # creates a 4-d and propagates the previously created index 2-d tensor for all batches
        indexes = indexes[None, :, :, None].repeat((batchSize, 1, 1, C)) # (batchSize, nW, windowSize, featureDim)

        src = rearrange(windowedX, "b n w c -> b (n w) c")
        indexes = rearrange(indexes, "b n w c -> b (n w) c")

        """
        scatter_add_() Adds all values from the tensor src into self at the indices specified in the index. For each 
        value in src, it is added to an index in self which is specified by its index in src for dimension != dim and 
        by the corresponding value in index for dimension = dim
        
        for a 3-d tensor: self[i][index[i][j][k]][k] += src[i][j][k]  # if dim == 1
        """

        destination.scatter_add_(dim=1, index=indexes, src=src)


        scalerSrc = torch.ones((windowLength)).to(device)[None, None, :, None].repeat(batchSize, nW, 1, C) # (batchSize, nW, windowLength, featureDim)
        scalerSrc = rearrange(scalerSrc, "b n w c -> b (n w) c")

        scalerDestination.scatter_add_(dim=1, index=indexes, src=scalerSrc)

        destination = destination / scalerDestination


        return destination



class BolTransformerBlock(nn.Module):

    """
    function:
        __init__() embeds the FusedWindowTransformer() as and attribute/function into the BolTransformerBlock module.
        Additionally, creates the masking indices when computing the context matrices of the attention heads


    inputs:
        dim - dimensions of the layer to normalize
        windowSize - number of sequential BOLD tokens in each window (excludes fringe tokens between adjacent windows)
        shiftSize - how many time units to skip when starting a subsequent time window
        receptiveSize - number of sequential BOLD tokens in receptive field (window tokens and fringe tokens)
        numHeads - number of attention heads in the FW-MSA (fused-window multi-attention head) mechanism
        headDim - number of output dimensions for each individual attention head in the FW-MSA mechanism
        mlpRatio - additional parameter to determine the output of the forward feed network
        attentionBias - True/False argument to add a learnable bias matrix to the attention matrix
        drop - probability that the content (projection) matrix's from FW-MSA mechanism elements are zeroed out
        attnDrop - probability that the attention matrix's elements are zeroed out

    outputs:
        embeds an object with the function FusedWindowTransformer() and masking attributes used to compute new output
        BOLD and CLS token embeddings. Once the input BOLD and CLS tokens pass through the fused window multi-attention
        head (FW-MSA) (via FusedWindowTransformer()) we will have new embedding representations for the BOLD and CLS
        tokens

    """

    def __init__(self, dim, numHeads, headDim, windowSize, receptiveSize, shiftSize, mlpRatio=1.0, drop=0.0, attnDrop=0.0, attentionBias=True):

        assert((receptiveSize-windowSize)%2 == 0)

        super().__init__()
        self.transformer = FusedWindowTransformer(dim=dim, windowSize=windowSize, shiftSize=shiftSize, receptiveSize=receptiveSize, numHeads=numHeads, headDim=headDim, mlpRatio=mlpRatio, attentionBias=attentionBias, drop=drop, attnDrop=attnDrop)

        self.windowSize = windowSize
        self.receptiveSize = receptiveSize
        self.shiftSize = shiftSize

        self.remainder = (self.receptiveSize - self.windowSize) // 2

        # create mask here for non matching query and key pairs
        maskCount = self.remainder // shiftSize + 1
        mask_left = torch.zeros(maskCount, self.windowSize+1, self.receptiveSize+1)
        mask_right = torch.zeros(maskCount, self.windowSize+1, self.receptiveSize+1)

        for i in range(maskCount):
            if(self.remainder > 0):
                mask_left[i, :, 1:1+self.remainder-shiftSize*i] = 1
                if(-self.remainder+shiftSize*i > 0):
                    mask_right[maskCount-1-i, :, -self.remainder+shiftSize*i:] = 1

        self.register_buffer("mask_left", mask_left)
        self.register_buffer("mask_right", mask_right)

    """
    function:
        getJuiceFlow(self) runs the getJuiceFlow function on the object that is it embedded in. This will produce a
        matrix that shows the average feature importance for a given context mapping (projection matrix) when 
        classifying on a particular class

    inputs:

    outputs:
        returns a matrix that shows average feature importance when classifying on a particular class

    """

    def getJuiceFlow(self):
        return self.transformer.getJuiceFlow()

    """
    function:
        forward(self, x, cls, analysis=False) takes a input of BOLD and CLS tokens and passes them through the fused
        window transformer block to capture the new embeddings of these tokens

    inputs:
        x : BOLD token tensor for a collection of subjects (batch). This input has dimension (batchSize, dynamicLength, c)
        cls : CLS token tensor for a collection of subjects (batch). This input has dimension (batchSize, nW, c)
        analysis : Boolean, it is set True only when you want to analyze the model, not important otherwise 

    outputs:
        fusedX_trans : fused window BOLD tokens with new embeddings after passing through the BolT fused window 
                       transformer block. This output is of dimension (batchSize, dynamicLength, c)
        cls_trans : fused window CLS tokens with new embeddings after passing through the BolT fused window 
                    transformer block. This output is of dimension(batchSize, nW, c)

    """
    
    def forward(self, x, cls, analysis=False):

        B, Z, C = x.shape
        device = x.device

        #update z, incase some are dropped during windowing
        Z = self.windowSize + self.shiftSize * (cls.shape[1]-1)
        x = x[:, :Z]

        # form the padded x to be used for focal keys and values
        x_ = torch.cat([torch.zeros((B, self.remainder,C),device=device), x, torch.zeros((B, self.remainder,C), device=device)], dim=1) # (B, remainder+Z+remainder, C) 

        # window the sequences
        windowedX, _ = windowBoldSignal(x.transpose(2,1), self.windowSize, self.shiftSize) # (B, nW, C, windowSize)         
        windowedX = windowedX.transpose(2,3) # (B, nW, windowSize, C)

        windowedX_, _ = windowBoldSignal(x_.transpose(2,1), self.receptiveSize, self.shiftSize) # (B, nW, C, receptiveSize)
        windowedX_ = windowedX_.transpose(2,3) # (B, nW, receptiveSize, C)

        
        nW = windowedX.shape[1] # number of windows
    
        xcls = torch.cat([cls.unsqueeze(dim=2), windowedX], dim = 2) # (B, nW, 1+windowSize, C)
        xcls = rearrange(xcls, "b nw l c -> (b nw) l c") # (B*nW, 1+windowSize, C) 
       
        xcls_ = torch.cat([cls.unsqueeze(dim=2), windowedX_], dim=2) # (B, nw, 1+receptiveSize, C)
        xcls_ = rearrange(xcls_, "b nw l c -> (b nw) l c") # (B*nW, 1+receptiveSize, C)

        masks = [self.mask_left, self.mask_right]

        # pass to fused window transformer
        fusedX_trans, cls_trans = self.transformer(x, cls, xcls, xcls_, masks, nW, analysis) # (B*nW, 1+windowSize, C)


        return fusedX_trans, cls_trans





