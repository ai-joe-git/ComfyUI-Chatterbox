"""
ComfyUI-Chatterbox: Text-to-Speech nodes for ComfyUI
Forced CPU inference (XPU has audio distortion issues)
"""

import torch

def apply_dtype_fix():
    """Fix Float/Double dtype mismatch in Chatterbox S3Tokenizer"""
    try:
        from chatterbox.models.s3tokenizer import s3tokenizer
        
        # S3_HOP is a module constant = 160 (for 100 frames/sec at 16kHz)
        S3_HOP = 160
        
        # Save original method
        original_log_mel = s3tokenizer.S3Tokenizer.log_mel_spectrogram
        
        def fixed_log_mel_spectrogram(self, audio, padding=0):
            """Fixed version - ensure float32 throughout"""
            import torch
            import torch.nn.functional as F
            
            if not torch.is_tensor(audio):
                audio = torch.from_numpy(audio)
            
            # FORCE float32 from the start
            audio = audio.to(self.device).to(torch.float32)
            
            if padding > 0:
                audio = F.pad(audio, (0, padding))
            
            # Use self.n_fft (exists as attribute) and S3_HOP constant
            stft = torch.stft(
                audio, 
                self.n_fft,      # This IS an attribute (= 400)
                S3_HOP,          # This is a module constant (= 160)
                window=self.window.to(self.device).to(torch.float32),
                return_complex=True
            )
            
            magnitudes = stft[..., :-1].abs()**2
            
            # Ensure mel_filters is float32
            mel_spec = self._mel_filters.to(self.device).to(torch.float32) @ magnitudes
            
            log_spec = torch.clamp(mel_spec, min=1e-10).log10()
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
            log_spec = (log_spec + 4.0) / 4.0
            
            # ENSURE OUTPUT IS FLOAT32
            return log_spec.to(torch.float32)
        
        # Apply the fix
        s3tokenizer.S3Tokenizer.log_mel_spectrogram = fixed_log_mel_spectrogram
        print("✅ Chatterbox dtype fix applied (float32 enforced)")
        
    except ImportError:
        print("⚠️  Chatterbox not installed yet")
    except Exception as e:
        print(f"⚠️  Could not apply dtype fix: {e}")

apply_dtype_fix()

from .chatterbox_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
