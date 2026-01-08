"""
ComfyUI-Chatterbox: Text-to-Speech nodes for ComfyUI
Supports Intel Arc XPU and CPU inference
"""

import os
import sys
import torch

# Apply XPU compatibility fixes before importing chatterbox
def apply_xpu_fixes():
    """Apply runtime fixes for Intel XPU compatibility"""
    try:
        # Import the original S3Tokenizer
        from chatterbox.models.s3tokenizer import s3tokenizer
        
        # Store original method
        original_log_mel = s3tokenizer.S3Tokenizer.log_mel_spectrogram
        
        def fixed_log_mel_spectrogram(self, wav):
            """Fixed version with XPU dtype compatibility"""
            # Compute STFT
            spec = torch.stft(
                wav,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window.to(wav.device),
                center=True,
                return_complex=True
            )
            
            # Get magnitudes
            magnitudes = spec.abs()
            
            # FIX: Ensure mel_filters matches magnitudes dtype (XPU requires exact match)
            mel_filters = self._mel_filters.to(self.device).to(magnitudes.dtype)
            mel_spec = mel_filters @ magnitudes
            
            # Log scale
            log_mel = torch.log(torch.clamp(mel_spec, min=1e-5))
            
            return log_mel
        
        # Apply the fix
        s3tokenizer.S3Tokenizer.log_mel_spectrogram = fixed_log_mel_spectrogram
        print("✅ XPU dtype compatibility applied")
        
    except ImportError:
        # Chatterbox not installed yet, will apply on first use
        pass
    except Exception as e:
        print(f"⚠️  Could not apply XPU fixes: {e}")

# Apply fixes at import time
apply_xpu_fixes()

# Import node classes
from .chatterbox_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
