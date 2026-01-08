"""
ComfyUI-Chatterbox: Text-to-Speech nodes for ComfyUI
Forced CPU inference (XPU has audio distortion issues)
"""

import os
import sys
import torch

# Apply dtype compatibility fixes
def apply_compatibility_fixes():
    """Apply runtime fixes for CPU/XPU compatibility"""
    try:
        from chatterbox.models.s3tokenizer import s3tokenizer
        
        original_log_mel = s3tokenizer.S3Tokenizer.log_mel_spectrogram
        
        def fixed_log_mel_spectrogram(self, wav):
            """Fixed version with dtype compatibility"""
            spec = torch.stft(
                wav,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window.to(wav.device),
                center=True,
                return_complex=True
            )
            
            magnitudes = spec.abs()
            
            # FIX: Ensure mel_filters matches magnitudes dtype
            mel_filters = self._mel_filters.to(self.device).to(magnitudes.dtype)
            mel_spec = mel_filters @ magnitudes
            
            log_mel = torch.log(torch.clamp(mel_spec, min=1e-5))
            
            return log_mel
        
        s3tokenizer.S3Tokenizer.log_mel_spectrogram = fixed_log_mel_spectrogram
        print("✅ Dtype compatibility applied")
        
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️  Could not apply fixes: {e}")

apply_compatibility_fixes()

from .chatterbox_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
