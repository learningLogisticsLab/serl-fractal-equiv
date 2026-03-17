import flax.linen as nn
import flax.numpy as np

from serl_launcher.vision.group_ops import C_4_1x1, C_4_3x3

class EquivariantEncoder(nn.Module):
    self.output_channels: Sequence[int] = (16,32,64,128)
    self.conv1: nn.Module = C_4_1x1(1,self.output_channels[0])
    self.convs: Sequence[nn.Module] = (C_4_3x3(self.output_channels[n-1], self.output_channels[n]) for n in range(1,len(output_channels)))
    self.pooling_strategy: str = "avg"
    self.stride: int = 1

    @nn.compact
    def __call__(self, obs):
        pool = nn.avg_pool
        
        obs = self.conv1(obs)
        obs = pool(obs, (2,2))
        for conv in self.convs:
            obs = conv(obs)
            obs = pool(obs,(2,2))
        
        obs = np.mean(obs, axis=(3,4)).reshape(obs.size(0),-1)
        
        return obs
