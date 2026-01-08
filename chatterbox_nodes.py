"""
ComfyUI-Chatterbox Nodes
Text-to-Speech using Resemble AI's Chatterbox models
Intel Arc XPU Compatible
"""

import os
import torch
import torchaudio
import numpy as np
import folder_paths
import tempfile
from pathlib import Path

# Audio reference folder
CHATTERBOX_AUDIO_DIR = os.path.join(folder_paths.base_path, "input", "chatterbox_audio")
os.makedirs(CHATTERBOX_AUDIO_DIR, exist_ok=True)

class ChatterboxTTSNode:
    """Main TTS node with voice cloning"""
    
    def __init__(self):
        self.model = None
        self.model_type_cache = None
        self.device = self._get_device()
    
    def _get_device(self):
        """Get best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            return "xpu"
        else:
            return "cpu"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello! This is Chatterbox."}),
                "model_type": (["turbo", "base", "multilingual"],),
                "language": (["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "ko", "hu", "hi"],),
                "exaggeration": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "cfg_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.1}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "normalize_volume": ("BOOLEAN", {"default": True}),
                "target_peak_db": ("FLOAT", {"default": -3.0, "min": -20.0, "max": 0.0, "step": 1.0}),
                "gain": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 5.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "reference_audio": ("AUDIO",),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate_speech"
    CATEGORY = "audio/chatterbox"
    
    def load_model(self, model_type):
        """Load Chatterbox model with XPU support"""
        
        # Check if already loaded
        if self.model is not None and self.model_type_cache == model_type:
            return self.model
        
        print(f"🎙️ Loading Chatterbox {model_type.title()}...")
        
        try:
            if model_type == "turbo":
                from chatterbox.tts_turbo import ChatterboxTurboTTS
                self.model = ChatterboxTurboTTS.from_pretrained(device=self.device)
                print("✅ Chatterbox Turbo (350M - 6x faster) loaded")
            elif model_type == "multilingual":
                from chatterbox.tts_multilingual import ChatterboxMultilingualTTS
                self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
                print("✅ Chatterbox Multilingual (23 languages) loaded")
            else:  # base
                from chatterbox.tts import ChatterboxTTS
                self.model = ChatterboxTTS.from_pretrained(device=self.device)
                print("✅ Chatterbox Base (0.5B - High Quality) loaded")
            
            self.model_type_cache = model_type
            print("   💡 Supports paralinguistic tags: [laugh], [chuckle], [sigh], [gasp], [cough]")
            return self.model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Chatterbox: {str(e)}\n\n"
                             "Installation:\n"
                             "  pip install chatterbox-tts\n"
                             "Or:\n"
                             "  pip install git+https://github.com/resemble-ai/chatterbox.git\n\n"
                             "If transformers error:\n"
                             "  pip install transformers==4.46.3 --force-reinstall")
    
    def process_audio(self, audio_path, target_sr=24000):
        """Load and process reference audio"""
        print("🎵 Processing reference audio...")
        
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        print(f"  ↳ Loaded: {waveform.shape[1]/sr:.2f}s @ {sr}Hz")
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            print("  ↳ Converted to mono")
        
        # Resample if needed
        if sr != target_sr:
            print(f"  ↳ Resampling {sr}Hz → {target_sr}Hz")
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        
        return waveform, target_sr
    
    def normalize_audio(self, waveform, target_peak_db=-3.0):
        """Normalize audio volume"""
        # Convert to numpy for processing
        audio_np = waveform.squeeze().numpy()
        
        # Get peak
        peak = np.abs(audio_np).max()
        if peak == 0:
            return waveform
        
        # Calculate gain to reach target peak
        target_peak = 10 ** (target_peak_db / 20)
        gain = target_peak / peak
        
        # Apply gain
        audio_np = audio_np * gain
        
        # Clip to prevent distortion
        audio_np = np.clip(audio_np, -1.0, 1.0)
        
        return torch.from_numpy(audio_np).unsqueeze(0)
    
    def generate_speech(self, text, model_type, language, exaggeration, cfg_weight, 
                       temperature, speed, normalize_volume, target_peak_db, gain, seed,
                       reference_audio=None):
        """Generate speech from text"""
        
        print(f"📝 Text ({len(text)} chars): '{text[:80]}...'")
        
        # Load model
        model = self.load_model(model_type)
        
        print(f"⚡ Using {model_type.title()} {'(6x faster than real-time)' if model_type == 'turbo' else ''}")
        
        # Prepare kwargs
        kwargs = {
            "exaggeration": float(exaggeration),
            "cfg_weight": float(cfg_weight),
            "temperature": float(temperature),
            "speed": float(speed),
            "seed": int(seed),
        }
        
        # Add language for multilingual
        if model_type == "multilingual":
            kwargs["language"] = language
        
        # Handle reference audio
        if reference_audio is not None:
            # Save reference audio to temp file
            ref_waveform = reference_audio["waveform"]
            ref_sr = reference_audio["sample_rate"]
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_path = tmp.name
                torchaudio.save(temp_path, ref_waveform, ref_sr)
            
            kwargs["audio_prompt_path"] = temp_path
        
        print("⚙️  " + ", ".join([f"{k}={v}" for k, v in list(kwargs.items())[:3]]))
        
        try:
            print("🎙️  Generating...")
            
            # Generate
            audio_output = model.generate(text, **kwargs)
            
            # Clean up temp file
            if reference_audio is not None:
                os.unlink(temp_path)
            
            # Process output
            if isinstance(audio_output, torch.Tensor):
                waveform = audio_output
            elif isinstance(audio_output, np.ndarray):
                waveform = torch.from_numpy(audio_output)
            else:
                raise ValueError(f"Unexpected audio output type: {type(audio_output)}")
            
            # Ensure correct shape [1, samples]
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.dim() == 3:
                waveform = waveform.squeeze(0)
            
            # Apply gain
            if gain != 1.0:
                waveform = waveform * gain
            
            # Normalize volume
            if normalize_volume:
                waveform = self.normalize_audio(waveform, target_peak_db)
            
            # Get sample rate from model
            sample_rate = 24000  # Chatterbox output rate
            
            duration = waveform.shape[1] / sample_rate
            print(f"✅ Generated {duration:.2f}s @ {sample_rate}Hz")
            
            return ({"waveform": waveform, "sample_rate": sample_rate},)
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            raise RuntimeError(f"Error generating speech: {str(e)}")


class ChatterboxLoadReferenceAudio:
    """Load reference audio for voice cloning"""
    
    @classmethod
    def INPUT_TYPES(cls):
        audio_files = [f for f in os.listdir(CHATTERBOX_AUDIO_DIR) 
                      if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))]
        
        if not audio_files:
            audio_files = ["No audio files found"]
        
        return {
            "required": {
                "audio_file": (audio_files,),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "load_audio"
    CATEGORY = "audio/chatterbox"
    
    def load_audio(self, audio_file):
        """Load reference audio file"""
        
        if audio_file == "No audio files found":
            raise ValueError(f"No audio files in {CHATTERBOX_AUDIO_DIR}")
        
        audio_path = os.path.join(CHATTERBOX_AUDIO_DIR, audio_file)
        
        print(f"📂 Loading: {audio_file}")
        
        waveform, sr = torchaudio.load(audio_path)
        
        duration = waveform.shape[1] / sr
        print(f"  ↳ {duration:.2f}s @ {sr}Hz")
        
        return ({"waveform": waveform, "sample_rate": sr},)


class ChatterboxPresets:
    """Preset configurations for different voice styles"""
    
    PRESETS = {
        "balanced": {
            "exaggeration": 0.5,
            "cfg_weight": 0.5,
            "temperature": 0.8,
            "target_peak_db": -3.0,
            "gain": 1.5
        },
        "expressive": {
            "exaggeration": 0.8,
            "cfg_weight": 0.7,
            "temperature": 1.0,
            "target_peak_db": -3.0,
            "gain": 1.5
        },
        "neutral": {
            "exaggeration": 0.2,
            "cfg_weight": 0.3,
            "temperature": 0.6,
            "target_peak_db": -3.0,
            "gain": 1.5
        },
        "dramatic": {
            "exaggeration": 1.0,
            "cfg_weight": 0.9,
            "temperature": 1.2,
            "target_peak_db": -3.0,
            "gain": 2.0
        }
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (list(cls.PRESETS.keys()),),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("exaggeration", "cfg_weight", "temperature", "target_peak_db", "gain")
    FUNCTION = "get_preset"
    CATEGORY = "audio/chatterbox"
    
    def get_preset(self, preset):
        """Return preset values"""
        print(f"🎚️  Preset: {preset}")
        
        values = self.PRESETS[preset]
        return (
            values["exaggeration"],
            values["cfg_weight"],
            values["temperature"],
            values["target_peak_db"],
            values["gain"]
        )


class ChatterboxSaveAudio:
    """Save generated audio to file"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "chatterbox_audio"}),
                "format": (["wav", "mp3", "flac"],),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save_audio"
    OUTPUT_NODE = True
    CATEGORY = "audio/chatterbox"
    
    def save_audio(self, audio, filename_prefix, format):
        """Save audio to file"""
        
        output_dir = folder_paths.get_output_directory()
        
        # Generate filename
        counter = 1
        while True:
            filename = f"{filename_prefix}_{counter:04d}.{format}"
            filepath = os.path.join(output_dir, filename)
            if not os.path.exists(filepath):
                break
            counter += 1
        
        # Save audio
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        
        torchaudio.save(filepath, waveform, sample_rate, format=format)
        
        print(f"💾 Saved: {filename}")
        
        return ()


# Node mappings
NODE_CLASS_MAPPINGS = {
    "ChatterboxTTSNode": ChatterboxTTSNode,
    "ChatterboxLoadReferenceAudio": ChatterboxLoadReferenceAudio,
    "ChatterboxPresets": ChatterboxPresets,
    "ChatterboxSaveAudio": ChatterboxSaveAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatterboxTTSNode": "Chatterbox TTS",
    "ChatterboxLoadReferenceAudio": "Load Reference Audio (Chatterbox)",
    "ChatterboxPresets": "Chatterbox Presets",
    "ChatterboxSaveAudio": "Save Audio (Chatterbox)",
}
