"""
ComfyUI-Chatterbox: Text-to-Speech nodes for ComfyUI
Forced CPU inference (XPU has audio distortion bugs)
"""

import torch

def apply_dtype_fix():
    """Fix Float/Double dtype mismatch in Chatterbox S3Tokenizer and VoiceEncoder"""
    try:
        from chatterbox.models.s3tokenizer import s3tokenizer
        from chatterbox.models.voice_encoder import voice_encoder
        
        # S3_HOP is a module constant = 160 (for 100 frames/sec at 16kHz)
        S3_HOP = 160
        
        # ========== FIX 1: S3Tokenizer ==========
        original_s3_log_mel = s3tokenizer.S3Tokenizer.log_mel_spectrogram
        
        def fixed_s3_log_mel_spectrogram(self, audio, padding=0):
            """Fixed S3Tokenizer version - ensure float32 throughout"""
            import torch
            import torch.nn.functional as F
            
            if not torch.is_tensor(audio):
                audio = torch.from_numpy(audio)
            
            audio = audio.to(self.device).to(torch.float32)
            
            if padding > 0:
                audio = F.pad(audio, (0, padding))
            
            stft = torch.stft(
                audio, 
                self.n_fft,
                S3_HOP,
                window=self.window.to(self.device).to(torch.float32),
                return_complex=True
            )
            
            magnitudes = stft[..., :-1].abs()**2
            mel_spec = self._mel_filters.to(self.device).to(torch.float32) @ magnitudes
            
            log_spec = torch.clamp(mel_spec, min=1e-10).log10()
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
            log_spec = (log_spec + 4.0) / 4.0
            
            return log_spec.to(torch.float32)
        
        s3tokenizer.S3Tokenizer.log_mel_spectrogram = fixed_s3_log_mel_spectrogram
        
        # ========== FIX 2: VoiceEncoder mel spectrogram ==========
        original_ve_log_mel = voice_encoder.VoiceEncoder.log_mel_spectrogram
        
        def fixed_ve_log_mel_spectrogram(self, wav):
            """Fixed VoiceEncoder version - ensure float32 output"""
            # Call original and force float32
            result = original_ve_log_mel(self, wav)
            if isinstance(result, torch.Tensor):
                return result.to(torch.float32)
            return result
        
        voice_encoder.VoiceEncoder.log_mel_spectrogram = fixed_ve_log_mel_spectrogram
        
        # ========== FIX 3: VoiceEncoder inference ==========
        original_inference = voice_encoder.VoiceEncoder.inference
        
        def fixed_inference(self, mels, mel_lens=None, batch_size=64, **kwargs):
            """Fixed inference - force float32 input"""
            # Ensure mels are float32 before passing to LSTM
            if isinstance(mels, torch.Tensor):
                mels = mels.to(torch.float32)
            return original_inference(self, mels, mel_lens, batch_size, **kwargs)
        
        voice_encoder.VoiceEncoder.inference = fixed_inference
        
        print("✅ Chatterbox dtype fix applied (S3Tokenizer + VoiceEncoder + Inference)")
        
    except ImportError:
        print("⚠️  Chatterbox not installed yet")
    except Exception as e:
        print(f"⚠️  Could not apply dtype fix: {e}")

apply_dtype_fix()

from .chatterbox_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
