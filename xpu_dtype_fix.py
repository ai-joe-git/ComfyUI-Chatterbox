"""XPU dtype compatibility patch for Chatterbox"""
import torch

# Store original function
_original_mel_spec = None

def patched_log_mel_spectrogram(self, wav):
    """Fixed mel spectrogram with dtype consistency"""
    # Get the original magnitudes
    with torch.no_grad():
        # Convert wav to proper format
        if wav.dtype != torch.float32:
            wav = wav.to(torch.float32)
        
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
        
        # ✅ FIX: Ensure mel_filters matches magnitudes dtype
        mel_filters = self._mel_filters.to(self.device).to(magnitudes.dtype)
        mel_spec = mel_filters @ magnitudes
        
        # Log scale
        log_mel = torch.log(torch.clamp(mel_spec, min=1e-5))
        
    return log_mel

def apply_xpu_dtype_patch():
    """Apply XPU dtype fix to s3tokenizer"""
    try:
        from chatterbox.models.s3tokenizer.s3tokenizer import S3Tokenizer
        global _original_mel_spec
        _original_mel_spec = S3Tokenizer.log_mel_spectrogram
        S3Tokenizer.log_mel_spectrogram = patched_log_mel_spectrogram
        print("✅ XPU dtype patch applied to s3tokenizer")
    except Exception as e:
        print(f"⚠️  Could not apply XPU dtype patch: {e}")
