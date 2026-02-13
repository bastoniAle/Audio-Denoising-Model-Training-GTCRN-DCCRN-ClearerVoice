import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchaudio
from models.w_scripts import STFT_W


class CConv2d(nn.Module):
    """
    Class of complex valued convolutional layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.real_conv = nn.Conv2d(in_channels=self.in_channels,
                                   out_channels=self.out_channels,
                                   kernel_size=self.kernel_size,
                                   padding=self.padding,
                                   stride=self.stride)
        self.im_conv = nn.Conv2d(in_channels=self.in_channels,
                                 out_channels=self.out_channels,
                                 kernel_size=self.kernel_size,
                                 padding=self.padding,
                                 stride=self.stride)

        #Glorot initialization.
        nn.init.xavier_uniform_(self.real_conv.weight)
        nn.init.xavier_uniform_(self.im_conv.weight)

    def forward(self, x):
        x_real = x[..., 0]
        x_imag = x[..., 1]

        c_real = self.real_conv(x_real) - self.im_conv(x_imag)
        c_imag = self.im_conv(x_real) + self.real_conv(x_imag)

        output = torch.stack([c_real, c_imag], dim=-1)
        return output


class CConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding=0, padding=0):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        self.real_convt = nn.ConvTranspose2d(in_channels=self.in_channels,
                                             out_channels=self.out_channels,
                                             kernel_size=self.kernel_size,
                                             output_padding=self.output_padding,
                                             padding=self.padding,
                                             stride=self.stride)

        self.im_convt = nn.ConvTranspose2d(in_channels=self.in_channels,
                                           out_channels=self.out_channels,
                                           kernel_size=self.kernel_size,
                                           output_padding=self.output_padding,
                                           padding=self.padding,
                                           stride=self.stride)

        # Glorot initialization
        nn.init.xavier_uniform_(self.real_convt.weight)
        nn.init.xavier_uniform_(self.im_convt.weight)

    def forward(self, x):
        x_real = x[..., 0]
        x_imag = x[..., 1]

        ct_real = self.real_convt(x_real) - self.im_convt(x_imag)
        ct_imag = self.real_convt(x_real) + self.im_convt(x_imag)

        output = torch.stack([ct_real, ct_imag], dim=-1)
        return output

class CBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.real_b = nn.BatchNorm2d(num_features=self.num_features, eps=self.eps, momentum=self.momentum, affine=self.affine, track_running_stats=self.track_running_stats)
        self.im_b = nn.BatchNorm2d(num_features=self.num_features, eps=eps, momentum=self.momentum, affine=self.affine, track_running_stats=self.track_running_stats)

    def forward(self, x):
        x_real = x[..., 0]
        x_imag = x[..., 1]

        n_real = self.real_b(x_real)
        n_imag = self.im_b(x_imag)
        output = torch.stack([n_real, n_imag], dim=-1)
        return output


class Encoder(nn.Module):
    def __init__(self, filter_size=(7,5), stride_size=(2,2), in_channels=1, out_channels=45, padding=(0,0)):
        super().__init__()

        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

        self.cconv = CConv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride_size, padding=self.padding)
        self.cbn = CBatchNorm2d(num_features=self.out_channels)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):

        conved = self.cconv(x)
        normed = self.cbn(conved)
        acted = self.leaky_relu(normed)

        return acted

class Decoder(nn.Module):
    def __init__(self, filter_size=(7,5), stride_size=(2,2), in_channels=1, out_channels=45, padding=(0,0), output_padding=(0,0), last_layer=False):
        super().__init__()

        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.output_padding = output_padding
        self.last_layer = last_layer

        self.cconvt = CConvTranspose2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.filter_size, stride=self.stride_size, output_padding=self.output_padding, padding=self.padding)

        self.cbn = CBatchNorm2d(num_features=self.out_channels)

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        conved = self.cconvt(x)

        if not self.last_layer:
            normed = self.cbn(conved)
            output = self.leaky_relu(normed)
        else:
            m_phase = conved / (torch.abs(conved) + 1e-8)
            m_mag = torch.tanh(torch.abs(conved))
            output = m_phase * m_mag

        return output


class DCUnet10(nn.Module):
    def __init__(self, n_fft=64, hop_length=16):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

        self.downsample0 = Encoder(filter_size=(7, 5), stride_size=(2,2), in_channels=1, out_channels=45)
        self.downsample1 = Encoder(filter_size=(7, 5), stride_size=(2, 2), in_channels=45, out_channels=90)
        self.downsample2 = Encoder(filter_size=(5, 3), stride_size=(2, 2), in_channels=90, out_channels=90)
        self.downsample3 = Encoder(filter_size=(5, 3), stride_size=(2, 2), in_channels=90, out_channels=90)
        self.downsample4 = Encoder(filter_size=(5, 3), stride_size=(2, 1), in_channels=90, out_channels=90)

        self.upsample0 = Encoder(filter_size=(5,3), stride_size=(2, 1), in_channels=90, out_channels=90, padding=(0,0), output_padding=(1,0))
        self.upsample1 = Encoder(filter_size=(5, 3), stride_size=(2, 2), in_channels=180, out_channels=90,
                                 padding=(0, 0), output_padding=(1, 1))
        self.upsample2 = Encoder(filter_size=(5, 3), stride_size=(2, 2), in_channels=180, out_channels=90,
                                 padding=(0, 0), output_padding=(1, 0))
        self.upsample3 = Encoder(filter_size=(7, 5), stride_size=(2, 2), in_channels=180, out_channels=45,
                                 padding=(0, 0), output_padding=(1, 0))
        self.upsample4 = Encoder(filter_size=(7, 5), stride_size=(2, 2), in_channels=90, out_channels=1,
                                 padding=(0, 0), output_padding=(0, 0), last_layer=True)

    def forward(self, x, is_istft=True):
        d0 = self.downsample0(x)
        d1 = self.downsample1(d0)
        d2 = self.downsample2(d1)
        d3 = self.downsample3(d2)
        d4 = self.downsample4(d3)

        u0 = self.upsample0(d4)
        c0 = torch.cat((u0, d3), dim=1)

        u1 = self.upsample1(c0)
        c1 = torch.cat((u1, d2), dim=1)

        u2 = self.upsample2(c1)
        c2 = torch.cat((u2, d1), dim=1)

        u3 = self.upsample3(c2)
        c3 = torch.cat((u3, d0), dim=1)

        u4 = self.upsample4(c3)
        output = u4 * x
        if is_istft:
            output = torch.squeeze(output, dim=1)
            output = torch.istft(output, n_fft=self.n_fft, hop_length=self.hop_length, normalized=True)

        return output


class DCUNET_SE_48K(nn.Module):
    def __init__(self, args):
        super(DCUNET_SE_48K, self).__init__()
        self.model = DCUnet10(n_fft=args.fft_len, hop_length=args.win_inc)
        self.win_type = args.win_type
        self.freq_analysis = STFT_W(n_fft=args.fft_len, hop_length=args.win_inc, win_length=args.win_len, window=self.win_type)

    def forward(self, x, y=None):
        A_stft = self.freq_analysis.stft(x)
        out = self.model(A_stft, is_istft=False)
        out_time = self.freq_analysis.istft(out)
        y_out = None
        if y != None:
            y_out = self.freq_analysis.stft(y)
        return out_time, y_out, out