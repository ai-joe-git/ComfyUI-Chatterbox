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
            # Get parameters - S3Tokenizer stores them in config
            n_fft = getattr(self, 'n_fft', 1024)
            hop_length = getattr(self, 'hop_length', 256)
            win_length = getattr(self, 'win_length', 1024)
            
            # Get window tensor
            if hasattr(self, 'window'):
                window = self.window.to(wav.device)
            else:
                window = torch.hann_window(win_length).to(wav.device)
            
            spec = torch.stft(
                wav,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=True,
                return_complex=True
            )
            
            magnitudes = spec.abs()
            
            # FIX: Ensure mel_filters matches magnitudes dtype
            if hasattr(self, '_mel_filters'):
                mel_filters = self._mel_filters.to(self.device).to(magnitudes.dtype)
            elif hasattr(self, 'mel_filters'):
                mel_filters = self.mel_filters.to(self.device).to(magnitudes.dtype)
            else:
                # Fallback: create mel filters
                from torchaudio.transforms import MelScale
                mel_scale = MelScale(
                    n_mels=80,
                    sample_rate=16000,
                    f_min=0,
                    f_max=8000,
                    n_stft=n_fft // 2 + 1
                ).to(wav.device)
                mel_filters = mel_scale.fb.T.to(magnitudes.dtype)
            
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
