
# import torchaudio
# torchaudio.set_audio_backend("soundfile")

import math
import torch

import torch.nn as nn

from utils.utils import fft_frequencies
from utils.utils  import LogarithmicFilterbank

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Flatten(nn.Module):
    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)


class LogSpectrogram(nn.Module):
    def __init__(self, sr, fps, frame_size, min_rate=0.5, max_rate=2., aug_prob=0.5):
        super(LogSpectrogram, self).__init__()

        self.sr = sr
        self.fps = fps
        self.n_fft = frame_size
        self.hop_length = int(self.sr / self.fps)
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.aug_prob = aug_prob
        self.min_frames = 10

        fbank = LogarithmicFilterbank(fft_frequencies(self.n_fft // 2 + 1, self.sr),
                                      num_bands=12, fmin=60, fmax=6000, norm_filters=True, unique_filters=False)
        fbank = torch.from_numpy(fbank)
        # print(fbank.shape)
        phase_advance = torch.linspace(0, math.pi * self.hop_length, 1025)[..., None]


        self.register_buffer('window', torch.hann_window(self.n_fft))
        self.register_buffer('fbank', fbank)
        self.register_buffer('phase_advance', phase_advance)

    def forward(self, x, tempo_aug=False):

        assert isinstance(x, list)
        specs = []
        
        for inp_ in x:

            if inp_.dim() != 3:
                # compute stft if not already computed
                x_stft = torch.stft(inp_, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window,
                                    center=False, return_complex=False)
            else:
                # input is already an stft
                x_stft = inp_

    
            # apply tempo augmentation, ie. time-stretching, if given signal is long enough (at least 10 frames long)
            # if tempo_aug and x_stft.shape[1] >= self.min_frames:

            #     # if np.random.rand() < self.aug_prob:
            #     if random.random() < self.aug_prob:
            #         rate = random.uniform(self.min_rate, self.max_rate)
            #         x_stft = torchaudio.functional.phase_vocoder(x_stft, rate, self.phase_advance)

            # magnitude spectrogram
            # print(x_stft.shape)
            # tmp = x_stft.pow(2).sum(-1).sqrt().T
            x_spec = x_stft.pow(2).sum(-1).sqrt().T

            # apply logarithmic filterbank
            # print(torch.matmul(x_spec, self.fbank).shape, x_spec.shape, self.fbank.shape)
            x_spec = torch.log10(torch.matmul(x_spec, self.fbank) + 1)

            specs.append(x_spec)
        
    
        return specs


class TemporalBatchNorm(nn.Module):
    """
    Batch normalization of a (batch, channels, bands, time) tensor over all but
    the previous to last dimension (the frequency bands).
    """
    def __init__(self, num_bands, affine=True):
        super(TemporalBatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(num_bands, affine=affine)

    def forward(self, x):
        shape = x.shape
        # squash channels into the batch dimension
        x = x.reshape((-1,) + x.shape[-2:])

        # B x T x F -> B x F x T
        x = x.permute(0, 2, 1)
        # pass through 1D batch normalization
        x = self.bn(x)

        # B x F x T -> B x T x F
        x = x.permute(0, 2, 1)

        # restore squashed dimensions
        return x.reshape(shape)


class Conv(nn.Module):
    """adapted from https://github.com/ultralytics/yolov5/blob/master/models/common.py"""
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, groupnorm=False, activation=nn.ELU):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), bias=False)
        self.norm = nn.GroupNorm(1, c2) if groupnorm else nn.BatchNorm2d(c2)
        self.act = activation(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Focus(nn.Module):
    """taken from https://github.com/ultralytics/yolov5/blob/master/models/common.py"""
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, groupnorm=False, activation=nn.ELU):
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, groupnorm=groupnorm, activation=activation)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Concat(nn.Module):
    """taken from https://github.com/ultralytics/yolov5/blob/master/models/common.py"""
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class FiLMConv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, zdim=128, groupnorm=False, activation=nn.ELU):
        super(FiLMConv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), bias=False)
        self.norm = nn.GroupNorm(1, c2) if groupnorm else nn.BatchNorm2d(c2)
        self.act = activation(inplace=True)

        self.gamma = nn.Linear(zdim, c2)
        self.beta = nn.Linear(zdim, c2)

    def forward(self, x, z):

        x = self.norm(self.conv(x))
        gamma = self.gamma(z).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(z).unsqueeze(-1).unsqueeze(-1)

        x = gamma * x + beta

        return self.act(x)


class Bottleneck(nn.Module):

    """
    adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, base_width=64, groupnorm=False, activation=nn.ELU):
        super(Bottleneck, self).__init__()

        width = int(planes * (base_width / 64.))

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.norm1 = nn.GroupNorm(1, width) if groupnorm else nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        self.norm2 = nn.GroupNorm(1, width) if groupnorm else nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.norm3 = nn.GroupNorm(1, planes * self.expansion) if groupnorm else nn.BatchNorm2d(planes * self.expansion)
        self.activation = activation(inplace=True)

        # self.downsample = downsample
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                # conv1x1(inplanes, planes*self.expansion, stride),
                nn.Conv2d(inplanes, planes * self.expansion, kernel_size=(1, 1), stride=stride, bias=False),
                nn.GroupNorm(1, planes*self.expansion) if groupnorm else nn.BatchNorm2d(planes*self.expansion),
            )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, groupnorm=False, activation=nn.ELU):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1, groupnorm=groupnorm, activation=activation)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, groupnorm=groupnorm, activation=activation)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(C2FBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0, groupnorm=groupnorm, activation=activation) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
class C2FBottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, groupnorm=False, activation=nn.ELU):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1, groupnorm=groupnorm, activation=activation)
        self.cv2 = Conv(c_, c2, k[1], 1, groupnorm=groupnorm, activation=activation)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))