import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import sys
import numpy as np
from torch.nn.modules.loss import _Loss
from torch.nn.utils.rnn import pad_sequence
from utils.misc import power_compress, power_uncompress, stft, istft, batch_pesq

EPS = 1e-6

def loss_frcrn_se_16k(args, noisy_wav, clean_wav, out_list, device):
    pred_wav = out_list[1]
    pred_cmask = out_list[2]
    SiSNR_loss = loss_sisnr(clean_wav, pred_wav)
    CMask_MSE_Loss = loss_complex_mask(args, noisy_wav, clean_wav, pred_cmask, device)
    loss = CMask_MSE_Loss + SiSNR_loss
    return loss, CMask_MSE_Loss, SiSNR_loss

def loss_mossformer2_se_48k(args, noisy_wav, clean_wav, mask_list, device):
    noisy_stft = stft(noisy_wav, args) 
    clean_stft = stft(clean_wav, args) 
    return psm_loss(noisy_stft, clean_stft, mask_list, device)

def loss_mossformergan_se_16k(args, inputs, labels, out_list, c, discriminator, device):
   
    one_labels = torch.ones(args.batch_size).to(device) 
    labels = torch.transpose(labels, 0, 1)
    labels = torch.transpose(labels * c, 0, 1)

    pred_real, pred_imag = out_list[0].permute(0, 1, 3, 2), out_list[1].permute(0, 1, 3, 2)
    pred_mag = torch.sqrt(pred_real**2 + pred_imag**2)

    labels_spec = stft(labels, args, center=True)
    labels_spec = labels_spec.to(torch.float32)
    labels_spec = power_compress(labels_spec)

    labels_real = labels_spec[:, 0, :, :].unsqueeze(1)
    labels_imag = labels_spec[:, 1, :, :].unsqueeze(1)
    labels_mag = torch.sqrt(labels_real**2 + labels_imag**2)

    predict_fake_metric = discriminator(labels_mag, pred_mag)
    gen_loss_GAN = F.mse_loss(predict_fake_metric.flatten(), one_labels.float())

    loss_mag = F.mse_loss(pred_mag, labels_mag)
    loss_ri = F.mse_loss(pred_real, labels_real) + F.mse_loss(pred_imag, labels_imag)

    pred_spec_uncompress = power_uncompress(labels_real, pred_imag).squeeze(1)
    pred_audio = istft(pred_spec_uncompress, args)

    length = min(pred_audio.size(-1), labels.size(-1))
    pred_audio = pred_audio[...,:length]
    labels = labels[...,:length]
    time_loss = torch.mean(torch.abs(pred_audio - labels))

    loss = 0.1 * loss_ri + 0.9 * loss_mag + 0.2 * time_loss + 0.05 * gen_loss_GAN
    if torch.isnan(loss):
        print('train loss is nan, skip this batch!')
        return None, None

    pred_audio_list = list(pred_audio.detach().cpu().numpy())
    labels_audio_list = list(labels.cpu().numpy())
    pesq_score = batch_pesq(labels_audio_list, pred_audio_list)

    # The calculation of PESQ can be None due to silent part
    if pesq_score is not None:
        predict_enhance_metric = discriminator(labels_mag, pred_mag.detach())
        predict_max_metric = discriminator(labels_mag, labels_mag)
        discrim_loss_metric = F.mse_loss(predict_max_metric.flatten(), one_labels) + \
                              F.mse_loss(predict_enhance_metric.flatten(), pesq_score)
    else:
        discrim_loss_metric = torch.tensor([0.])
        print('train pesq score is None, skip this batch!')
        return None, None

    return loss, discrim_loss_metric

def cal_SISNR(source, estimate_source):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        source: torch tensor, [batch size, sequence length]
        estimate_source: torch tensor, [batch size, sequence length]
    Returns:
        SISNR, [batch size]
    """
    assert source.size() == estimate_source.size()
    # Step 1. Zero-mean norm
    source = source - torch.mean(source, axis = -1, keepdim=True)
    estimate_source = estimate_source - torch.mean(estimate_source, axis = -1, keepdim=True)
    # Step 2. SI-SNR
    # s_target = <s', s>s / ||s||^2
    ref_energy = torch.sum(source ** 2, axis = -1, keepdim=True) + EPS
    proj = torch.sum(source * estimate_source, axis = -1, keepdim=True) * source / ref_energy
    # e_noise = s' - s_target
    noise = estimate_source - proj
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    ratio = torch.sum(proj ** 2, axis = -1) / (torch.sum(noise ** 2, axis = -1) + EPS)
    sisnr = 10 * torch.log10(ratio + EPS)
    return sisnr

def psm_loss(noisy_stft, clean_stft, mask_list, device, ratio_mod="l2"):
    #masks: B x T x D
    #noisy_stft: B x D x T x 2
    noisy_abs = torch.sqrt(noisy_stft[..., 0] ** 2 + noisy_stft[..., 1] ** 2)
    noisy_abs_max, _ = torch.max(noisy_abs, dim=1, keepdim=True)
    noisy_abs_norm = noisy_abs / (noisy_abs_max + 1e-6)
    noisy_abs_norm  = noisy_abs_norm.permute(0, 2, 1)
    ##noisy_abs_norm: B x T x D
    noisy_real = noisy_stft[..., 0] / (noisy_abs + EPS)
    noisy_imag = noisy_stft[..., 1] / (noisy_abs + EPS)
    clean_abs = torch.sqrt(clean_stft[..., 0] ** 2 + clean_stft[..., 1] ** 2)
    clean_real = clean_stft[..., 0] / (clean_abs + EPS)
    clean_imag = clean_stft[..., 1] / (clean_abs + EPS)

    if ratio_mod == "l2":
        tgt_mask = clean_abs ** 2 / (noisy_abs ** 2 + EPS) * (noisy_real * clean_real + noisy_imag * clean_imag)
    elif ratio_mod == "l1":
        tgt_mask = clean_abs/ (noisy_abs  + EPS) * (noisy_real * clean_real + noisy_imag * clean_imag)

    tgt_mask = torch.clamp(tgt_mask, 0, 1)
    tgt_mask = tgt_mask.permute(0, 2, 1)    
    pred_mask = mask_list[-1]

    total_frames = tgt_mask.shape[1] * tgt_mask.shape[0]
    loss = 0.5*torch.sum(torch.pow(pred_mask - tgt_mask, 2) * noisy_abs_norm)/total_frames
    
    if len(mask_list) > 1:
        for i in range(len(mask_list)-1):
            pred_mask = mask_list[i]
            loss = loss + 0.5*torch.sum(torch.pow(pred_mask - tgt_mask, 2)* noisy_abs_norm)/total_frames
    
    return loss

def loss_sisnr(clean_wav, est_wav):
    if clean_wav.dim() == 3:
        clean_wav = torch.squeeze(clean_wav,1)
    if est_wav.dim() == 3:
        est_wav = torch.squeeze(est_wav,1)
    sisnr = cal_SISNR(clean_wav, est_wav)
    return -torch.mean(sisnr)

def loss_complex_mask(args, noisy_wav, clean_wav, est_cmask, device):
    fft_bins = args.fft_len // 2 +1
    S = stft(clean_wav, args) 
    Sr = S[...,0]
    Si = S[...,1]
    Y = stft(noisy_wav, args)
    Yr = Y[...,0]
    Yi = Y[...,1]
    Y_pow = Yr**2 + Yi**2
    Y_mag = torch.sqrt(Y_pow)
    gth_mask = torch.cat([(Sr*Yr+Si*Yi)/(Y_pow + 1e-8),(Si*Yr-Sr*Yi)/(Y_pow + 1e-8)], 1)
    gth_mask[gth_mask > 2] = 1
    gth_mask[gth_mask < -2] = -1
    mask_real_loss = F.mse_loss(gth_mask[:,:fft_bins, :], est_cmask[:,:fft_bins, :]) * args.fft_len
    mask_imag_loss = F.mse_loss(gth_mask[:,fft_bins:, :], est_cmask[:,fft_bins:, :]) * args.fft_len
    CMask_MSE_Loss = mask_real_loss + mask_imag_loss
    return CMask_MSE_Loss

def loss_dccrn_wsdr_fn(x_, y_pred_, y_true_, eps=1e-8):
    if y_true_.dim() == 3:
        y_true_ = torch.squeeze(y_true_, 1)
    if y_pred_.dim() == 3:
        y_pred_ = torch.squeeze(y_pred_, 1)
    if x_.dim() == 3:
        x_ = torch.squeeze(x_, 1)

    y_pred = y_pred_.flatten(1)
    y_true = y_true_.flatten(1)
    x = x_.flatten(1)

    def sdr_fn(true, pred, eps=1e-8):
        num = torch.sum(true * pred, dim=1)
        den = torch.norm(true, p=2, dim=1) * torch.norm(pred, p=2, dim=1)
        return -(num / (den + eps))

    # true and estimated noise
    z_true = x - y_true
    z_pred = x - y_pred
    a = torch.sum(y_true**2, dim=1) / (torch.sum(y_true**2, dim=1) + torch.sum(z_true**2, dim=1) + eps)
    wSDR = a * sdr_fn(y_true, y_pred) + (1 - a) * sdr(z_true, z_pred)
    return torch.mean(wSDR)

# dpcrn loss
def dpcrn_spec_loss(y_true_re, y_true_im, y_pred_re, y_pred_im):
    mse_fn = nn.MSELoss()
    # real-part loss
    real_loss = mse_fn(y_true_re, y_pred_re)
    # imag-part loss
    imag_loss = mse_fn(y_true_im, y_pred_im)
    # magnitude loss
    y_true_mag = (y_true_re ** 2 + y_true_im ** 2 + 1e-8) ** 0.5
    y_pred_mag = (y_pred_re ** 2 + y_pred_im ** 2 + 1e-8) ** 0.5
    mag_loss = mse_fn(y_true_mag, y_pred_mag)

    total_loss = mag_loss + real_loss + imag_loss
    total_loss = torch.log(total_loss + 1e-8)
    return total_loss

def dpcrn_snr_loss(y_true, y_pred):
    snr = torch.mean(torch.square(y_true), dim=-1, keepdim=True) / torch.mean(torch.square(y_pred - y_true), dim=-1, keepdim=True + 1e-8)
    loss = -10 * torch.log10(snr + 1e-8)
    loss = torch.squeeze(loss, -1).mean()
    return loss

def DPCRNLoss(y_true, y_pred, y_true_re, y_true_im, y_pred_re, y_pred_im):
    loss1 = dpcrn_snr_loss(y_true, y_pred)
    loss2 = dpcrn_spec_loss(y_true_re, y_true_im, y_pred_re, y_pred_im)
    loss = loss1 + loss2
    return loss, loss1, loss2

class gtcrnHybridLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_stft, true_stft, frame_len, frame_inc, fft_len, window):
        device = pred_stft.device

        pred_stft_real, pred_stft_imag = pred_stft[:,:,:,0], pred_stft[:,:,:,1]
        true_stft_real, true_stft_imag = true_stft[:,:,:,0], true_stft[:,:,:,1]
        pred_mag = torch.sqrt(pred_stft_real**2 + pred_stft_imag**2 + 1e-12)
        true_mag = torch.sqrt(true_stft_real**2 + true_stft_imag**2 + 1e-12)
        pred_real_c = pred_stft_real / (pred_mag ** (0.7))
        pred_imag_c = pred_stft_imag / (pred_mag ** (0.7))
        true_real_c = true_stft_real / (true_mag ** (0.7))
        true_imag_c = true_stft_imag / (true_mag ** (0.7))
        real_loss = nn.MSELoss()(pred_real_c, true_real_c)
        imag_loss = nn.MSELoss()(pred_imag_c, true_imag_c)
        mag_loss = nn.MSELoss()(pred_mag**(0.3), true_mag**(0.3))

        y_pred = torch.istft(pred_stft_real+1j*pred_stft_imag, fft_len, frame_inc, frame_len, window=window)
        y_true = torch.istft(true_stft_real+1j*true_stft_imag, fft_len, frame_inc, frame_len, window=window)
        y_true = torch.sum(y_true * y_pred, dim=-1, keepdim=True) * y_true / (torch.sum(torch.square(y_true), dim=-1, keepdim=True) + 1e-12)

        sisnr = -torch.log10(torch.norm(y_true, dim=-1, keepdim=True) ** 2 / (torch.norm(y_pred - y_true, dim=-1, keepdim=True)**2+1e-8) + 1e-8).mean()

        return 30*(real_loss + imag_loss) + 70*mag_loss + sisnr, sisnr