
#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
#Author: Shengkui Zhao

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch 
import torch.nn as nn
import numpy as np
import os 
import sys
import librosa
import torchaudio
from utils.misc import power_compress, power_uncompress, stft, istft, compute_fbank
MAX_WAV_VALUE = 32768.0

def decode_one_audio(model, device, inputs, args):
    if args.network == 'FRCRN_SE_16K':
        return decode_one_audio_frcrn_se_16k(model, device, inputs, args)
    elif args.network == 'MossFormer2_SE_48K':
        return decode_one_audio_mossformer2_se_48k(model, device, inputs, args)
    elif args.network == 'MossFormerGAN_SE_16K':
        return decode_one_audio_mossformergan_se_16k(model, device, inputs, args)
    elif args.network == 'DCUNET_SE_48K':
        return decode_one_audio_dunet_se_48k(model, device, inputs, args)
    elif args.network == 'DCCRN_SE_48K':
        return decode_one_audio_dccrn_se_48k(model, device, inputs, args)
    elif args.network == 'PHASEN_SE_48K':
        return decode_one_audio_phasen_se_48k(model, device, inputs, args)
    elif args.network == 'DPCRN_SE_48K':
        return decode_one_audio_dpcrn_se_48k(model, device, inputs, args)
    elif args.network == 'GTCRN_SE_48K' or args.network == 'ULUNAS_SE_48K':
        return decode_one_audio_gtcrn_se_48k(model, device, inputs, args)
    elif args.network == 'DTLN_SE_48K':
        return decode_one_audio_dtln_se_48k(model, device, inputs, args)
    else:
       print("No network found!")
       return

def decode_one_audio_dcunet_se_48k(model, device, inputs, args):
    decode_do_segement = False
    max_length = args.max_length * args.sampling_rate
    window = args.sampling_rate * args.decode_window
    stride = int(window * 0.75)
    b, t = inputs.shape
    if t > args.sampling_rate * args.one_time_decode_length:
        decode_do_segement = True

    # Pad input to meet window/stride requirements
    if t < window:
        inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], window - t))], 1)
    elif t < window + stride:
        padding = window + stride - t
        inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], padding))], 1)
    else:
        if (t - window) % stride != 0:
            padding = t - (t - window) // stride * stride
            inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], padding))], 1)

    inputs = torch.from_numpy(np.float32(inputs))
    inputs = inputs.to(device)
    b, t = inputs.shape
    if decode_do_segement:
        outputs = np.zeros(t, dtype=np.float32)
        give_up_length = (window - stride) // 2
        current_idx = 0

        while current_idx + window <= t:
            tmp_input = inputs[:, current_idx:current_idx + window]

            if tmp_input.size(1) < max_length:
                num_padding = max_length - tmp_input.size(1)
                inputs_A = torch.nn.functional.pad(tmp_input, (0, num_padding))
            else:
                inputs_A = tmp_input

            tmp_output, _, _ = model(inputs_A)
            tmp_output = tmp_output.cpu().numpy()

            # === CRITICAL FIX: Squeeze to (batch, time) ===
            if tmp_output.ndim > 2:
                tmp_output = tmp_output.reshape(tmp_output.shape[0], -1)
            elif tmp_output.ndim == 1:
                tmp_output = tmp_output[np.newaxis, :]
            # ===============================================

            actual_output_len = tmp_output.shape[1]
            target_len = window

            if actual_output_len > target_len:
                aligned_output = tmp_output[:, :target_len]
            elif actual_output_len < target_len:
                aligned_output = np.pad(tmp_output, ((0, 0), (0, target_len - actual_output_len)), mode='constant')
            else:
                aligned_output = tmp_output

            aligned_output = aligned_output.reshape(-1)

            if current_idx == 0:
                segment = aligned_output[:-give_up_length]
                outputs[current_idx:current_idx + len(segment)] = segment
            else:
                segment = aligned_output[give_up_length:-give_up_length]
                start_idx = current_idx + give_up_length
                end_idx = start_idx + len(segment)
                outputs[start_idx:end_idx] = segment

            current_idx += stride
    else:
        if inputs.size(1) < max_length:
            num_padding = max_length - inputs.size(1)
            inputs_A = torch.nn.functional.pad(inputs, (0, num_padding))
        else:
            inputs_A = inputs

        outputs, _, _ = model(inputs_A)
        outputs = outputs.cpu().numpy()

        if outputs.ndim > 2:
            outputs = outputs.reshape(outputs.shape[0], -1)
        elif outputs.ndim == 1:
            outputs = outputs[np.newaxis, :]

        actual_len = outputs.shape[1]
        target_len = t
        if actual_len > target_len:
            outputs = outputs[:, :target_len]
        elif actual_len < target_len:
            outputs = np.pad(outputs, ((0, 0), (0, target_len - actual_len)), mode='constant')

        outputs = outputs.reshape(-1)

    return outputs


def decode_one_audio_phasen_se_48k(model, device, inputs, args):
    decode_do_segement = False
    max_length = args.max_length * args.sampling_rate
    window = args.sampling_rate * args.decode_window
    stride = int(window * 0.75)
    b, t = inputs.shape
    if t > args.sampling_rate * args.one_time_decode_length:
        decode_do_segement = True

    # Pad input to meet window/stride requirements
    if t < window:
        inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], window - t))], 1)
    elif t < window + stride:
        padding = window + stride - t
        inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], padding))], 1)
    else:
        if (t - window) % stride != 0:
            padding = t - (t - window) // stride * stride
            inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], padding))], 1)

    inputs = torch.from_numpy(np.float32(inputs))
    inputs = inputs.to(device)
    b, t = inputs.shape
    if decode_do_segement:
        outputs = np.zeros(t, dtype=np.float32)
        give_up_length = (window - stride) // 2
        current_idx = 0

        while current_idx + window <= t:
            tmp_input = inputs[:, current_idx:current_idx + window]

            if tmp_input.size(1) < max_length:
                num_padding = max_length - tmp_input.size(1)
                inputs_A = torch.nn.functional.pad(tmp_input, (0, num_padding))
            else:
                inputs_A = tmp_input

            _, tmp_output = model(inputs_A)
            tmp_output = tmp_output.cpu().numpy()

            # === CRITICAL FIX: Squeeze to (batch, time) ===
            if tmp_output.ndim > 2:
                tmp_output = tmp_output.reshape(tmp_output.shape[0], -1)
            elif tmp_output.ndim == 1:
                tmp_output = tmp_output[np.newaxis, :]
            # ===============================================

            actual_output_len = tmp_output.shape[1]
            target_len = window

            if actual_output_len > target_len:
                aligned_output = tmp_output[:, :target_len]
            elif actual_output_len < target_len:
                aligned_output = np.pad(tmp_output, ((0, 0), (0, target_len - actual_output_len)), mode='constant')
            else:
                aligned_output = tmp_output

            aligned_output = aligned_output.reshape(-1)

            if current_idx == 0:
                segment = aligned_output[:-give_up_length]
                outputs[current_idx:current_idx + len(segment)] = segment
            else:
                segment = aligned_output[give_up_length:-give_up_length]
                start_idx = current_idx + give_up_length
                end_idx = start_idx + len(segment)
                outputs[start_idx:end_idx] = segment

            current_idx += stride
    else:
        if inputs.size(1) < max_length:
            num_padding = max_length - inputs.size(1)
            inputs_A = torch.nn.functional.pad(inputs, (0, num_padding))
        else:
            inputs_A = inputs

        _, outputs = model(inputs_A)
        outputs = outputs.cpu().numpy()

        if outputs.ndim > 2:
            outputs = outputs.reshape(outputs.shape[0], -1)
        elif outputs.ndim == 1:
            outputs = outputs[np.newaxis, :]

        actual_len = outputs.shape[1]
        target_len = t
        if actual_len > target_len:
            outputs = outputs[:, :target_len]
        elif actual_len < target_len:
            outputs = np.pad(outputs, ((0, 0), (0, target_len - actual_len)), mode='constant')

        outputs = outputs.reshape(-1)

    return outputs


def decode_one_audio_dccrn_se_48k(model, device, inputs, args):
    decode_do_segement = False
    max_length = args.max_length * args.sampling_rate
    window = args.sampling_rate * args.decode_window
    stride = int(window * 0.75)
    b, t = inputs.shape
    if t > args.sampling_rate * args.one_time_decode_length:
        decode_do_segement = True

    # Pad input to meet window/stride requirements
    if t < window:
        inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], window - t))], 1)
    elif t < window + stride:
        padding = window + stride - t
        inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], padding))], 1)
    else:
        if (t - window) % stride != 0:
            padding = t - (t - window) // stride * stride
            inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], padding))], 1)

    inputs = torch.from_numpy(np.float32(inputs))
    inputs = inputs.to(device)
    b, t = inputs.shape
    if decode_do_segement:
        outputs = np.zeros(t, dtype=np.float32)
        give_up_length = (window - stride) // 2
        current_idx = 0

        while current_idx + window <= t:
            tmp_input = inputs[:, current_idx:current_idx + window]

            if tmp_input.size(1) < max_length:
                num_padding = max_length - tmp_input.size(1)
                inputs_A = torch.nn.functional.pad(tmp_input, (0, num_padding))
            else:
                inputs_A = tmp_input

            _, _, tmp_output = model(inputs_A)
            tmp_output = tmp_output.cpu().numpy()

            # === CRITICAL FIX: Squeeze to (batch, time) ===
            if tmp_output.ndim > 2:
                tmp_output = tmp_output.reshape(tmp_output.shape[0], -1)
            elif tmp_output.ndim == 1:
                tmp_output = tmp_output[np.newaxis, :]
            # ===============================================

            actual_output_len = tmp_output.shape[1]
            target_len = window

            if actual_output_len > target_len:
                aligned_output = tmp_output[:, :target_len]
            elif actual_output_len < target_len:
                aligned_output = np.pad(tmp_output, ((0, 0), (0, target_len - actual_output_len)), mode='constant')
            else:
                aligned_output = tmp_output

            aligned_output = aligned_output.reshape(-1)

            if current_idx == 0:
                segment = aligned_output[:-give_up_length]
                outputs[current_idx:current_idx + len(segment)] = segment
            else:
                segment = aligned_output[give_up_length:-give_up_length]
                start_idx = current_idx + give_up_length
                end_idx = start_idx + len(segment)
                outputs[start_idx:end_idx] = segment

            current_idx += stride
    else:
        if inputs.size(1) < max_length:
            num_padding = max_length - inputs.size(1)
            inputs_A = torch.nn.functional.pad(inputs, (0, num_padding))
        else:
            inputs_A = inputs

        _, _, outputs = model(inputs_A)
        outputs = outputs.cpu().numpy()

        if outputs.ndim > 2:
            outputs = outputs.reshape(outputs.shape[0], -1)
        elif outputs.ndim == 1:
            outputs = outputs[np.newaxis, :]

        actual_len = outputs.shape[1]
        target_len = t
        if actual_len > target_len:
            outputs = outputs[:, :target_len]
        elif actual_len < target_len:
            outputs = np.pad(outputs, ((0, 0), (0, target_len - actual_len)), mode='constant')

        outputs = outputs.reshape(-1)

    return outputs

def decode_one_audio_dpcrn_se_48k(model, device, inputs, args):
    decode_do_segement = False
    max_length = args.max_length * args.sampling_rate
    window = args.sampling_rate * args.decode_window
    stride = int(window * 0.75)
    b, t = inputs.shape
    if t > args.sampling_rate * args.one_time_decode_length:
        decode_do_segement = True

    # Pad input to meet window/stride requirements
    if t < window:
        inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], window - t))], 1)
    elif t < window + stride:
        padding = window + stride - t
        inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], padding))], 1)
    else:
        if (t - window) % stride != 0:
            padding = t - (t - window) // stride * stride
            inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], padding))], 1)

    inputs = torch.from_numpy(np.float32(inputs))
    inputs = inputs.to(device)
    b, t = inputs.shape
    if decode_do_segement:
        outputs = np.zeros(t, dtype=np.float32)
        give_up_length = (window - stride) // 2
        current_idx = 0

        while current_idx + window <= t:
            tmp_input = inputs[:, current_idx:current_idx + window]

            if tmp_input.size(1) < max_length:
                num_padding = max_length - tmp_input.size(1)
                inputs_A = torch.nn.functional.pad(tmp_input, (0, num_padding))
            else:
                inputs_A = tmp_input

            tmp_output, _, _ = model(inputs_A)
            tmp_output = tmp_output.cpu().numpy()

            # === CRITICAL FIX: Squeeze to (batch, time) ===
            if tmp_output.ndim > 2:
                tmp_output = tmp_output.reshape(tmp_output.shape[0], -1)
            elif tmp_output.ndim == 1:
                tmp_output = tmp_output[np.newaxis, :]
            # ===============================================

            actual_output_len = tmp_output.shape[1]
            target_len = window

            if actual_output_len > target_len:
                aligned_output = tmp_output[:, :target_len]
            elif actual_output_len < target_len:
                aligned_output = np.pad(tmp_output, ((0, 0), (0, target_len - actual_output_len)), mode='constant')
            else:
                aligned_output = tmp_output

            aligned_output = aligned_output.reshape(-1)

            if current_idx == 0:
                segment = aligned_output[:-give_up_length]
                outputs[current_idx:current_idx + len(segment)] = segment
            else:
                segment = aligned_output[give_up_length:-give_up_length]
                start_idx = current_idx + give_up_length
                end_idx = start_idx + len(segment)
                outputs[start_idx:end_idx] = segment

            current_idx += stride
    else:
        if inputs.size(1) < max_length:
            num_padding = max_length - inputs.size(1)
            inputs_A = torch.nn.functional.pad(inputs, (0, num_padding))
        else:
            inputs_A = inputs

        outputs, _, _ = model(inputs_A)
        outputs = outputs.cpu().numpy()

        if outputs.ndim > 2:
            outputs = outputs.reshape(outputs.shape[0], -1)
        elif outputs.ndim == 1:
            outputs = outputs[np.newaxis, :]

        actual_len = outputs.shape[1]
        target_len = t
        if actual_len > target_len:
            outputs = outputs[:, :target_len]
        elif actual_len < target_len:
            outputs = np.pad(outputs, ((0, 0), (0, target_len - actual_len)), mode='constant')

        outputs = outputs.reshape(-1)

    return outputs


def decode_one_audio_gtcrn_se_48k(model, device, inputs, args):
    decode_do_segement = False
    max_length = args.max_length * args.sampling_rate
    window = args.sampling_rate * args.decode_window
    stride = int(window * 0.75)
    b, t = inputs.shape
    if t > args.sampling_rate * args.one_time_decode_length:
        decode_do_segement = True

    # Pad input to meet window/stride requirements
    if t < window:
        inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], window - t))], 1)
    elif t < window + stride:
        padding = window + stride - t
        inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], padding))], 1)
    else:
        if (t - window) % stride != 0:
            padding = t - (t - window) // stride * stride
            inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], padding))], 1)

    inputs = torch.from_numpy(np.float32(inputs))
    inputs = inputs.to(device)
    b, t = inputs.shape
    if decode_do_segement:
        outputs = np.zeros(t, dtype=np.float32)
        give_up_length = (window - stride) // 2
        current_idx = 0

        while current_idx + window <= t:
            tmp_input = inputs[:, current_idx:current_idx + window]

            if tmp_input.size(1) < max_length:
                num_padding = max_length - tmp_input.size(1)
                inputs_A = torch.nn.functional.pad(tmp_input, (0, num_padding))
            else:
                inputs_A = tmp_input

            tmp_output, _, _ = model(inputs_A)
            tmp_output = tmp_output.cpu().numpy()

            # === CRITICAL FIX: Squeeze to (batch, time) ===
            if tmp_output.ndim > 2:
                tmp_output = tmp_output.reshape(tmp_output.shape[0], -1)
            elif tmp_output.ndim == 1:
                tmp_output = tmp_output[np.newaxis, :]
            # ===============================================

            actual_output_len = tmp_output.shape[1]
            target_len = window

            if actual_output_len > target_len:
                aligned_output = tmp_output[:, :target_len]
            elif actual_output_len < target_len:
                aligned_output = np.pad(tmp_output, ((0, 0), (0, target_len - actual_output_len)), mode='constant')
            else:
                aligned_output = tmp_output

            aligned_output = aligned_output.reshape(-1)

            if current_idx == 0:
                segment = aligned_output[:-give_up_length]
                outputs[current_idx:current_idx + len(segment)] = segment
            else:
                segment = aligned_output[give_up_length:-give_up_length]
                start_idx = current_idx + give_up_length
                end_idx = start_idx + len(segment)
                outputs[start_idx:end_idx] = segment

            current_idx += stride
    else:
        if inputs.size(1) < max_length:
            num_padding = max_length - inputs.size(1)
            inputs_A = torch.nn.functional.pad(inputs, (0, num_padding))
        else:
            inputs_A = inputs

        outputs, _, _ = model(inputs_A)
        outputs = outputs.cpu().numpy()

        if outputs.ndim > 2:
            outputs = outputs.reshape(outputs.shape[0], -1)
        elif outputs.ndim == 1:
            outputs = outputs[np.newaxis, :]

        actual_len = outputs.shape[1]
        target_len = t
        if actual_len > target_len:
            outputs = outputs[:, :target_len]
        elif actual_len < target_len:
            outputs = np.pad(outputs, ((0, 0), (0, target_len - actual_len)), mode='constant')

        outputs = outputs.reshape(-1)

    return outputs

def decode_one_audio_dtln_se_48k(model, device, inputs, args):
    decode_do_segement = False
    max_length = args.max_length *args.sampling_rate
    window = args.sampling_rate * args.decode_window
    stride = int(window*0.75)
    b,t = inputs.shape
    if t > args.sampling_rate * args.one_time_decode_length:
        decode_do_segement=True

    # Pad input to meet window/stride requirements
    if t < window:
        inputs = np.concatenate([inputs,np.zeros((inputs.shape[0],window-t))],1)
    elif t < window + stride:
        padding = window + stride - t
        inputs = np.concatenate([inputs,np.zeros((inputs.shape[0],padding))],1)
    else:
        if (t - window) % stride != 0:
            padding = t - (t-window)//stride * stride
            inputs = np.concatenate([inputs,np.zeros((inputs.shape[0],padding))],1)

    inputs = torch.from_numpy(np.float32(inputs))
    inputs = inputs.to(device)
    b, t = inputs.shape
    if decode_do_segement:
        outputs = np.zeros(t, dtype=np.float32)
        give_up_length = (window - stride) // 2
        current_idx = 0

        while current_idx + window <= t:
            tmp_input = inputs[:, current_idx:current_idx+window]

            if tmp_input.size(1) < max_length:
                num_padding = max_length - tmp_input.size(1)
                inputs_A = torch.nn.functional.pad(tmp_input, (0, num_padding))
            else:
                inputs_A = tmp_input

            tmp_output = model(inputs_A)
            tmp_output = tmp_output.cpu().numpy()

            # === CRITICAL FIX: Squeeze to (batch, time) ===
            if tmp_output.ndim > 2:
                tmp_output = tmp_output.reshape(tmp_output.shape[0], -1)
            elif tmp_output.ndim == 1:
                tmp_output = tmp_output[np.newaxis, :]
            # ===============================================

            actual_output_len = tmp_output.shape[1]
            target_len = window

            if actual_output_len > target_len:
                aligned_output = tmp_output[:, :target_len]
            elif actual_output_len < target_len:
                aligned_output = np.pad(tmp_output, ((0, 0), (0, target_len - actual_output_len)), mode='constant')
            else:
                aligned_output = tmp_output

            aligned_output = aligned_output.reshape(-1)

            if current_idx == 0:
                segment = aligned_output[:-give_up_length]
                outputs[current_idx:current_idx + len(segment)] = segment
            else:
                segment = aligned_output[give_up_length:-give_up_length]
                start_idx = current_idx + give_up_length
                end_idx = start_idx + len(segment)
                outputs[start_idx:end_idx] = segment

            current_idx += stride
    else:
        if inputs.size(1) < max_length:
            num_padding = max_length - inputs.size(1)
            inputs_A = torch.nn.functional.pad(inputs, (0, num_padding))
        else:
            inputs_A = inputs

        outputs = model(inputs_A)
        outputs = outputs.cpu().numpy()

        if outputs.ndim > 2:
            outputs = outputs.reshape(outputs.shape[0], -1)
        elif outputs.ndim == 1:
            outputs = outputs[np.newaxis, :]

        actual_len = outputs.shape[1]
        target_len = t
        if actual_len > target_len:
            outputs = outputs[:, :target_len]
        elif actual_len < target_len:
            outputs = np.pad(outputs, ((0, 0), (0, target_len - actual_len)), mode='constant')

        outputs = outputs.reshape(-1)

    return outputs




def decode_one_audio_frcrn_se_16k(model, device, inputs, args):

    decode_do_segement=False

    window = args.sampling_rate * args.decode_window  #decoding window length 16000 # 1s
    stride = int(window*0.75)
    b,t = inputs.shape

    if t > args.sampling_rate * args.one_time_decode_length:
        decode_do_segement=True
    if t < window:
        inputs = np.concatenate([inputs,np.zeros((inputs.shape[0],window-t))],1)
    elif t < window + stride:
        padding = window + stride - t
        inputs = np.concatenate([inputs,np.zeros((inputs.shape[0],padding))],1)
    else:
        if (t - window) % stride != 0:
            padding = t - (t-window)//stride * stride
            inputs = np.concatenate([inputs,np.zeros((inputs.shape[0],padding))],1)

    inputs = torch.from_numpy(np.float32(inputs))
    inputs = inputs.to(device)
    b,t = inputs.shape
    if decode_do_segement:
        outputs = np.zeros(t)
        give_up_length=(window - stride)//2
        current_idx = 0
        while current_idx + window <= t:
            tmp_input = inputs[:,current_idx:current_idx+window]
            tmp_output = model.inference(tmp_input,).cpu().numpy()
            if current_idx == 0:
                outputs[current_idx:current_idx+window-give_up_length] = tmp_output[:-give_up_length]

            else:
                outputs[current_idx+give_up_length:current_idx+window-give_up_length] = tmp_output[give_up_length:-give_up_length]
            current_idx += stride
    else:
        outputs = model.inference(inputs,).cpu().numpy()

    return outputs

def decode_one_audio_mossformergan_se_16k(model, device, inputs, args):

    decode_do_segement=False
    window = args.sampling_rate * args.decode_window #16000 # 1s
    stride = int(window*0.75)
    b,t = inputs.shape
    if t > args.sampling_rate * args.one_time_decode_length:
        decode_do_segement=True

    if t < window:
        inputs = np.concatenate([inputs,np.zeros((inputs.shape[0],window-t))],1)
    elif t < window + stride:
        padding = window + stride - t
        inputs = np.concatenate([inputs,np.zeros((inputs.shape[0],padding))],1)
    else:
        if (t - window) % stride != 0:
            padding = t - (t-window)//stride * stride
            inputs = np.concatenate([inputs,np.zeros((inputs.shape[0],padding))],1)
    inputs = torch.from_numpy(np.float32(inputs))
    inputs = inputs.to(device)
    b,t = inputs.shape
    if decode_do_segement:
        outputs = np.zeros(t)
        give_up_length=(window - stride)//2
        current_idx = 0
        while current_idx + window <= t:
            tmp_input = inputs[:,current_idx:current_idx+window]
            tmp_output = _decode_one_audio_mossformergan_se_16k(model, device, tmp_input, args)
            if current_idx == 0:
                outputs[current_idx:current_idx+window-give_up_length] = tmp_output[:-give_up_length]

            else:
                outputs[current_idx+give_up_length:current_idx+window-give_up_length] = tmp_output[give_up_length:-give_up_length]
            current_idx += stride
        return outputs
    else:
        return _decode_one_audio_mossformergan_se_16k(model, device, inputs, args)

def _decode_one_audio_mossformergan_se_16k(model, device, inputs, args):
    input_len = inputs.size(-1)
    nframe = int(np.ceil(input_len / args.win_inc))
    padded_len = nframe * args.win_inc
    padding_len = padded_len - input_len
    inputs = torch.cat([inputs, inputs[:, :padding_len]], dim=-1)

    c = torch.sqrt(inputs.size(-1) / torch.sum((inputs ** 2.0), dim=-1))
    inputs = torch.transpose(inputs, 0, 1)
    inputs = torch.transpose(inputs * c, 0, 1)
    inputs_spec = stft(inputs, args, center=True)
    inputs_spec = inputs_spec.to(torch.float32)
    inputs_spec = power_compress(inputs_spec).permute(0, 1, 3, 2)
    out_list = model(inputs_spec)
    pred_real, pred_imag = out_list[0].permute(0, 1, 3, 2), out_list[1].permute(0, 1, 3, 2)
    pred_spec_uncompress = power_uncompress(pred_real, pred_imag).squeeze(1)
    outputs = istft(pred_spec_uncompress, args)
    outputs = outputs.squeeze(0) / c
    outputs = outputs[:input_len]
    return outputs.detach().cpu().numpy()

def decode_one_audio_mossformer2_se_48k(model, device, inputs, args):
    inputs = inputs[0,:]
    input_len = inputs.shape[0]
    inputs = inputs * MAX_WAV_VALUE
    if input_len > args.sampling_rate * args.one_time_decode_length: ## longer than 20s, use online decoding
        online_decoding = True
        if online_decoding:
            window = int(args.sampling_rate * args.decode_window) ## 4s window for 48kHz sample rate
            stride = int(window * 0.75) ## 3s stride for 48kHz sample rate
            t = inputs.shape[0]

            if t < window:
                inputs = np.concatenate([inputs,np.zeros(window-t)],0)
            elif t < window + stride:
                padding = window + stride - t
                inputs = np.concatenate([inputs,np.zeros(padding)],0)
            else:
                if (t - window) % stride != 0:
                    padding = t - (t-window)//stride * stride
                    inputs = np.concatenate([inputs,np.zeros(padding)],0)
            audio = torch.from_numpy(inputs).type(torch.FloatTensor)
            t = audio.shape[0]
            outputs = torch.from_numpy(np.zeros(t))
            give_up_length=(window - stride)//2
            dfsmn_memory_length = 0 
            current_idx = 0
            while current_idx + window <= t:
                if current_idx < dfsmn_memory_length:
                    audio_segment = audio[0:current_idx+window]
                else:
                    audio_segment = audio[current_idx-dfsmn_memory_length:current_idx+window]
                fbanks = compute_fbank(audio_segment.unsqueeze(0), args)
                # compute deltas for fbank
                fbank_tr = torch.transpose(fbanks, 0, 1)
                fbank_delta = torchaudio.functional.compute_deltas(fbank_tr)
                fbank_delta_delta = torchaudio.functional.compute_deltas(fbank_delta)
                fbank_delta = torch.transpose(fbank_delta, 0, 1)
                fbank_delta_delta = torch.transpose(fbank_delta_delta, 0, 1)
                fbanks = torch.cat([fbanks, fbank_delta, fbank_delta_delta], dim=1)

                fbanks =fbanks.unsqueeze(0).to(device)
                Out_List = model(fbanks)
                pred_mask = Out_List[-1]
                spectrum = stft(audio_segment, args)
                pred_mask = pred_mask.permute(2,1,0)
                masked_spec = spectrum.cpu() * pred_mask.detach().cpu()
                masked_spec_complex = masked_spec[:,:,0] + 1j*masked_spec[:,:,1]
                output_segment = istft(masked_spec_complex, args, len(audio_segment))
                if current_idx == 0:
                    outputs[current_idx:current_idx+window-give_up_length] = output_segment[:-give_up_length]
                else:
                    output_segment = output_segment[-window:]
                    outputs[current_idx+give_up_length:current_idx+window-give_up_length] = output_segment[give_up_length:-give_up_length]
                current_idx += stride
    else:
        audio = torch.from_numpy(inputs).type(torch.FloatTensor)
        fbanks = compute_fbank(audio.unsqueeze(0), args)
        # compute deltas for fbank
        fbank_tr = torch.transpose(fbanks, 0, 1)
        fbank_delta = torchaudio.functional.compute_deltas(fbank_tr)
        fbank_delta_delta = torchaudio.functional.compute_deltas(fbank_delta)
        fbank_delta = torch.transpose(fbank_delta, 0, 1)
        fbank_delta_delta = torch.transpose(fbank_delta_delta, 0, 1)
        fbanks = torch.cat([fbanks, fbank_delta, fbank_delta_delta], dim=1)

        fbanks =fbanks.unsqueeze(0).to(device)

        Out_List = model(fbanks)
        pred_mask = Out_List[-1]
        spectrum = stft(audio, args)
        pred_mask = pred_mask.permute(2,1,0) 
        masked_spec = spectrum * pred_mask.detach().cpu()
        masked_spec_complex = masked_spec[:,:,0] + 1j*masked_spec[:,:,1]
        outputs = istft(masked_spec_complex, args, len(audio))

    return outputs.numpy() / MAX_WAV_VALUE
