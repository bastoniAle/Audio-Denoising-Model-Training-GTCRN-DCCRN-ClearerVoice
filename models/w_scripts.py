import torch

class STFT_W:
    def __init__(self, n_fft=1024, hop_length=None, win_length=None, window=None, device=torch.device('cuda')):
        self.n_fft = n_fft
        self.length = None
        if win_length is None:
            self.win_length = self.n_fft
        else:
            self.win_length = win_length
        if hop_length is None:
            self.hop_length = self.n_fft // 2
        else:
            self.hop_length = hop_length
        if window is None or window == 'hann' or window == 'hanning':
            self.window = torch.hann_window(self.win_length).to(device)
        elif window == 'hamming':
            self.window = torch.hamming_window(self.win_length).to(device)

    def stft(self, data, no_channel=True):
        length = data.shape[-1]
        self.length = length
        if len(data.shape) == 1:
            data = torch.reshape(data, (1, length))

        data_stft = torch.stft(data, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window, return_complex=True)
        data_out = torch.stack((data_stft.real, data_stft.imag), dim=-1)
        B, Freq, Frame, Com = data_out.shape
        data_out = data_out.view(B, 1, Freq, Frame, Com)
        return data_out

    def istft(self, data, no_channel=True):
        Batch, Channel, Freqs, Frame, Complex = data.shape
        if no_channel:
            data = data.view(Batch, Freqs, Frame, Complex)
        data_stft = torch.complex(data[...,0], data[...,1])
        data_time = torch.istft(data_stft, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window, length=self.length)
        return data_time

if __name__ == '__main__':
    temp = torch.randn(3, 160000).to('cuda')
    freq_analysis = STFT_W(n_fft=1024, hop_length=512, win_length=1024, window='hann')
    stft = freq_analysis.stft
    istft = freq_analysis.istft

    A = stft(temp)
    B = istft(A)
    print(B.shape)
    print(temp.shape)