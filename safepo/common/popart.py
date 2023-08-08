
import numpy as np
import torch
import torch.nn as nn


class PopArt(nn.Module):
    """
    PopArt normalization module.

    This class implements PopArt normalization, which normalizes a vector of observations across the specified dimensions.

    Args:
        input_shape (tuple): Shape of the input vector.
        norm_axes (int): Number of dimensions along which normalization is performed (default: 1).
        beta (float): Exponential moving average decay factor (default: 0.99999).
        per_element_update (bool): Flag indicating whether to perform per-element update (default: False).
        epsilon (float): Small value to avoid division by zero (default: 1e-5).
        device (torch.device): Device to use for computations (default: cpu).

    Attributes:
        input_shape (tuple): Shape of the input vector.
        norm_axes (int): Number of dimensions along which normalization is performed.
        epsilon (float): Small value to avoid division by zero.
        beta (float): Exponential moving average decay factor.
        per_element_update (bool): Flag indicating whether to perform per-element update.
        tpdv (dict): Dictionary for tensor properties.
        running_mean (nn.Parameter): Running mean of input values.
        running_mean_sq (nn.Parameter): Running mean of squared input values.
        debiasing_term (nn.Parameter): Debiasing term for mean and variance computation.

    Methods:
        reset_parameters(): Reset running statistics and debiasing term to initial values.
        running_mean_var(): Compute debiased running mean and variance.
        forward(input_vector, train=True): Perform forward pass through the normalization module.
        denormalize(input_vector): Transform normalized data back into the original distribution.

    Example:
        input_shape = (64, 128)
        popart = PopArt(input_shape)
        normalized_data = torch.randn(10, *input_shape)
        denormalized_data = popart.denormalize(normalized_data)
    """

    def __init__(self, input_shape, norm_axes=1, beta=0.99999, per_element_update=False, epsilon=1e-5, device=torch.device("cpu")):
        super(PopArt, self).__init__()

        self.input_shape = input_shape
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.per_element_update = per_element_update
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.running_mean = nn.Parameter(torch.zeros(input_shape), requires_grad=False).to(**self.tpdv)
        self.running_mean_sq = nn.Parameter(torch.zeros(input_shape), requires_grad=False).to(**self.tpdv)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False).to(**self.tpdv)

    def reset_parameters(self):
        """Reset running statistics and debiasing term to initial values."""
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        """
        Compute debiased running mean and variance.

        Returns:
            tuple: Tuple containing debiased running mean and variance tensors.
        """
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(min=self.epsilon)
        debiased_var = (debiased_mean_sq - debiased_mean ** 2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    def forward(self, input_vector, train=True):
        """
        Perform a forward pass through the normalization module.

        Args:
            input_vector (torch.Tensor or np.ndarray): Input vector to normalize.
            train (bool): Flag indicating whether to update running statistics (default: True).

        Returns:
            torch.Tensor: Normalized output tensor.
        """
        # Make sure input is float32
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        if train:

            detached_input = input_vector.detach()
            batch_mean = detached_input.mean(dim=tuple(range(self.norm_axes)))
            batch_sq_mean = (detached_input ** 2).mean(dim=tuple(range(self.norm_axes)))

            if self.per_element_update:
                batch_size = np.prod(detached_input.size()[:self.norm_axes])
                weight = self.beta ** batch_size
            else:
                weight = self.beta

            self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
            self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
            self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

        mean, var = self.running_mean_var()
        out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[(None,) * self.norm_axes]
        
        return out

    def denormalize(self, input_vector):
        """
        Transform normalized data back into the original distribution.

        Args:
            input_vector (torch.Tensor or np.ndarray): Normalized input vector to denormalize.

        Returns:
            torch.Tensor: Denormalized output tensor.
        """
        if type(input_vector) == np.ndarray:
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var)[(None,) * self.norm_axes] + mean[(None,) * self.norm_axes]
        
        out = out.detach()
        
        return out
