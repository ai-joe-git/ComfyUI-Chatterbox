"""
ComfyUI-Chatterbox: Text-to-Speech nodes for ComfyUI
Forced CPU inference (XPU has audio distortion issues)
"""

import torch

def apply_dtype_fix():
    """Fix Float/Double dtype mismatch in Chatterbox"""
    try:
        from chatterbox.models.s3tokenizer import s3tokenizer
        
        # Save original method
        original_log_mel = s3tokenizer.S3Tokenizer.log_mel_spectrogram
        
        def fixed_log_mel_spectrogram(self, wav):
            """Fixed version that ensures dtype compatibility"""
            # Call original but fix the dtype issue
            import torch
            
            # Get STFT parameters
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
            
            # FIX: Ensure mel_filters matches magnitudes dtype (both float32)
            mel_filters = self._mel_filters.to(self.device).to(magnitudes.dtype)
            mel_spec = mel_filters @ magnitudes
            
            log_mel = torch.log(torch.clamp(mel_spec, min=1e-5))
            
            return log_mel
        
        # Apply the fix
        s3tokenizer.S3Tokenizer.log_mel_spectrogram = fixed_log_mel_spectrogram
        print("✅ Chatterbox dtype fix applied")
        
    except ImportError:
        print("⚠️  Chatterbox not installed yet")
    except Exception as e:
        print(f"⚠️  Could not apply dtype fix: {e}")

apply_dtype_fix()

from .chatterbox_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
