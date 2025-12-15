import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HaarWavelet2D(nn.Module):
    """
    Implements the Fully Separable Wavelet Transform (FSWT) using Haar wavelets.
    Directly ported from SWM_simplified.py
    """
    def __init__(self):
        super().__init__()
        s2 = 1.0 / np.sqrt(2.0)
        self.register_buffer('h_L', torch.tensor([[[[s2, s2]]]], dtype=torch.float32))
        self.register_buffer('h_H', torch.tensor([[[[-s2, s2]]]], dtype=torch.float32))
        self.register_buffer('h_L_inv', torch.tensor([[[[s2, s2]]]], dtype=torch.float32))
        self.register_buffer('h_H_inv', torch.tensor([[[[-s2, s2]]]], dtype=torch.float32))

    def _get_kernels(self, x):
        return (
            self.h_L.to(device=x.device, dtype=x.dtype),
            self.h_H.to(device=x.device, dtype=x.dtype),
            self.h_L_inv.to(device=x.device, dtype=x.dtype),
            self.h_H_inv.to(device=x.device, dtype=x.dtype)
        )

    def decomposition_step(self, x):
        # Expects 4D input: (Batch*Time, C, H, W)
        B, C, H, W = x.shape
        x = x.reshape(B * C, 1, H, W)
        h_L, h_H, _, _ = self._get_kernels(x)

        L_row = F.conv2d(x, h_L, stride=(1, 2), padding=0)
        H_row = F.conv2d(x, h_H, stride=(1, 2), padding=0)
        
        # Transpose to convolve columns
        L_row_t, H_row_t = L_row.transpose(2, 3), H_row.transpose(2, 3)

        LL = F.conv2d(L_row_t, h_L, stride=(1, 2)).transpose(2, 3)
        LH = F.conv2d(L_row_t, h_H, stride=(1, 2)).transpose(2, 3)
        HL = F.conv2d(H_row_t, h_L, stride=(1, 2)).transpose(2, 3)
        HH = F.conv2d(H_row_t, h_H, stride=(1, 2)).transpose(2, 3)

        return (
            LL.reshape(B, C, LL.shape[2], LL.shape[3]),
            LH.reshape(B, C, LH.shape[2], LH.shape[3]),
            HL.reshape(B, C, HL.shape[2], HL.shape[3]),
            HH.reshape(B, C, HH.shape[2], HH.shape[3])
        )

    def reconstruction_step(self, LL, LH, HL, HH):
        B, C, H, W = LL.shape
        _, _, h_L_inv, h_H_inv = self._get_kernels(LL)

        LL = LL.reshape(B * C, 1, H, W)
        LH = LH.reshape(B * C, 1, H, W)
        HL = HL.reshape(B * C, 1, H, W)
        HH = HH.reshape(B * C, 1, H, W)

        LL_t, LH_t = LL.transpose(2, 3), LH.transpose(2, 3)
        HL_t, HH_t = HL.transpose(2, 3), HH.transpose(2, 3)

        L_row_t = F.conv_transpose2d(LL_t, h_L_inv, stride=(1, 2)) + \
                  F.conv_transpose2d(LH_t, h_H_inv, stride=(1, 2))
        H_row_t = F.conv_transpose2d(HL_t, h_L_inv, stride=(1, 2)) + \
                  F.conv_transpose2d(HH_t, h_H_inv, stride=(1, 2))

        L_row, H_row = L_row_t.transpose(2, 3), H_row_t.transpose(2, 3)
        x = F.conv_transpose2d(L_row, h_L_inv, stride=(1, 2)) + \
            F.conv_transpose2d(H_row, h_H_inv, stride=(1, 2))

        return x.reshape(B, C, x.shape[2], x.shape[3])

def apply_waverep(real_clip, fake_clip, wavelet_processor, mode='full'):
    # 1. Align dimensions
    C, T, H, W = real_clip.shape
    if fake_clip.shape != real_clip.shape:
        fake_clip = F.interpolate(fake_clip.permute(1,0,2,3), size=(H,W), mode='bilinear').permute(1,0,2,3)

    real_2d = real_clip.permute(1, 0, 2, 3).reshape(T, C, H, W)
    fake_2d = fake_clip.permute(1, 0, 2, 3).reshape(T, C, H, W)

    # 2. Decomposition (L=3)
    # Real
    r_LL1, r_LH1, r_HL1, r_HH1 = wavelet_processor.decomposition_step(real_2d)
    r_LL2, r_LH2, r_HL2, r_HH2 = wavelet_processor.decomposition_step(r_LL1)
    r_LL3, r_LH3, r_HL3, r_HH3 = wavelet_processor.decomposition_step(r_LL2)
    # Fake
    f_LL1, f_LH1, f_HL1, f_HH1 = wavelet_processor.decomposition_step(fake_2d)
    f_LL2, f_LH2, f_HL2, f_HH2 = wavelet_processor.decomposition_step(f_LL1)
    f_LL3, f_LH3, f_HL3, f_HH3 = wavelet_processor.decomposition_step(f_LL2)

    # This ensures HH (Diagonal) and any other non-replaced bands remain Fake
    m_LL3 = f_LL3
    m_LH3, m_LH2, m_LH1 = f_LH3, f_LH2, f_LH1
    m_HL3, m_HL2, m_HL1 = f_HL3, f_HL2, f_HL1
    m_HH3, m_HH2, m_HH1 = f_HH3, f_HH2, f_HH1 
    # -------------------------------------------------------------

    # 3. Replacement Logic (SWM Core)
    # The paper states: "replace the low-frequency bands (plus those along the 
    # horizontal and vertical directions) with those of the real counterpart" 
    
    if mode == 'base':
        # Only replace the lowest frequency approximation
        m_LL3 = r_LL3 
        
    elif mode == 'horizontal':
        # Replace Base + Horizontal details (LH bands detect Vertical edges, HL detect Horizontal)
        # Assuming you want to replace "H" and "V" components:
        m_LL3 = r_LL3
        m_LH3, m_LH2, m_LH1 = r_LH3, r_LH2, r_LH1
        m_HL3, m_HL2, m_HL1 = r_HL3, r_HL2, r_HL1

    elif mode == 'full': 
        # The SWM Strategy: Replace EVERYTHING except Diagonals (HH)
        m_LL3 = r_LL3
        # Replace Vertical-sensitive bands (LH) with Real
        m_LH3, m_LH2, m_LH1 = r_LH3, r_LH2, r_LH1
        # Replace Horizontal-sensitive bands (HL) with Real
        m_HL3, m_HL2, m_HL1 = r_HL3, r_HL2, r_HL1
        # m_HH (Diagonal) remains FAKE (from initialization)

    # 4. Reconstruction
    rec_LL2 = wavelet_processor.reconstruction_step(m_LL3, m_LH3, m_HL3, m_HH3)
    rec_LL1 = wavelet_processor.reconstruction_step(rec_LL2, m_LH2, m_HL2, m_HH2)
    aug_2d  = wavelet_processor.reconstruction_step(rec_LL1, m_LH1, m_HL1, m_HH1)

    return aug_2d.reshape(T, C, H, W).permute(1, 0, 2, 3)
