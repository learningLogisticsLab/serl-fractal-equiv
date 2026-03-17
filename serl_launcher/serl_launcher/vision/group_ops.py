import flax.linen as nn
import flax
import jax
import jax.random
import jax.numpy as np
import time
import math


'''
Defines a 1x1 kernel convolutional layer where out_channels is the number of features to output.
'''
class C_4_1x1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(C_4_1x1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        key = jax.random.key(time.time_ns())
        weight = jax.random.normal(key, (in_channels, out_channels, 4)) / math.sqrt(2 * in_channels)

        self.weight = self.param("weight", lambda : weight)
        self.stride = stride
        self.pool = nn.max_pool(2,2)
    def __call__(self, x):
        # build weight tensor
        weight = np.zeros((self.out_channels, 4, self.in_channels, 4))

        weight = weight.at[:, 0, ...].set(self.weight)
        weight = weight.at[:, 1, ...].set(self.weight[..., [3, 0, 1, 2]])
        weight = weight.at[:, 2, ...].set(self.weight[..., [2, 3, 0, 1]])
        weight = weight.at[:, 3, ...].set(self.weight[..., [1, 2, 3, 0]])

        weight = weight.reshape(self.out_channels * 4, self.in_channels * 4, 1, 1)

        # convert to HWIO for JAX conv
        weight = np.transpose(weight, (2, 3, 1, 0))

        x = jax.lax.conv_general_dilated(
            x,
            weight,
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )

        if self.stride != 1:
            x = nn.avg_pool(x, window_shape=(self.stride, self.stride), strides=(self.stride, self.stride))

        return x

class C_4_3x3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(C_4_1x1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        key = jax.random.key(time.time_ns())
        weight = jax.random.normal(key, (in_channels, out_channels, 4)) / math.sqrt(2 * in_channels)

        self.weight = self.param("weight", lambda : weight)
        self.stride = stride
        self.pool = nn.max_pool(2,2)
    def __call__(self, x):
        weight = np.zeros((self.out_channels, 4, self.in_channels, 4, 3, 3))

        weight[::,0,...] = self.weight
        weight[::,1,...] = np.rot90(self.weight[...,[3,0,1,2],::,::], 1, [3,4])
        weight[::,2,...] = np.rot90(self.weight[...,[3,0,1,2],::,::], 2, [3,4])
        weight[::,3,...] = np.rot90(self.weight[...,[3,0,1,2],::,::], 3, [3,4])
        
        weight = weight.reshape(self.out_channels * 4, self.in_channels * 4, 3, 3)

        x = jax.lax.conv_general_dilated(
            x,
            weight,
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )

        if self.stride != 1:
            x = nn.avg_pool(x, window_shape=(self.stride, self.stride), strides=(self.stride, self.stride))

        return x
