"""
ComfyUI-Chatterbox Nodes
Text-to-Speech using Resemble AI's Chatterbox models
FORCED CPU INFERENCE (XPU has audio distortion bugs)
✅ Fixed: ComfyUI standard audio format [batch, channels, samples]
"""

import os
import torch
import torchaudio
import soundfile as sf
import numpy as np
import folder_paths
import tempfile
from pathlib import Path

# Audio reference folder
CHATTERBOX_AUDIO_DIR = os.path.join(folder_paths.base_path, "input", "chatterbox_audio")
os.makedirs(CHATTERBOX_AUDIO_DIR, exist_ok=True)

class ChatterboxTTSNode:
    """Main TTS node with voice cloning (CPU-only)"""
    
    def __init__(self):
        self.model = None
        self.model_type_cache = None
        self.device = "cpu"  # FORCE CPU - XPU causes audio distortion
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello! This is Chatterbox."}),
                "model_type": (["turbo", "base", "multilingual"],),
                "language": (["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "ko", "hu", "hi"],),
                "exaggeration": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.1}),
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
        """Load Chatterbox model (CPU-only)"""
        
        if self.model is not None and self.model_type_cache == model_type:
            return self.model
        
        print(f"🎙️  Loading Chatterbox {model_type.title()} on CPU...")
        print("   ℹ️  CPU inference (XPU disabled due to audio distortion bugs)")
        
        try:
            original_load = torch.load
            
            def cpu_load(f, *args, **kwargs):
                if 'map_location' not in kwargs:
                    kwargs['map_location'] = 'cpu'
                return original_load(f, *args, **kwargs)
            
            torch.load = cpu_load
            
            try:
                if model_type == "turbo":
                    from chatterbox.tts_turbo import ChatterboxTurboTTS
                    self.model = ChatterboxTurboTTS.from_pretrained(device="cpu")
                    print("✅ Chatterbox Turbo (350M - 6x faster) loaded on CPU")
                    
                elif model_type == "multilingual":
                    from chatterbox.tts_multilingual import ChatterboxMultilingualTTS
                    self.model = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
                    print("✅ Chatterbox Multilingual (23 languages) loaded on CPU")
                    
                else:  # base
                    from chatterbox.tts import ChatterboxTTS
                    self.model = ChatterboxTTS.from_pretrained(device="cpu")
                    print("✅ Chatterbox Base (0.5B - High Quality) loaded on CPU")
                
            finally:
                torch.load = original_load
            
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
    
    def normalize_audio(self, waveform, target_peak_db=-3.0):
        """Normalize audio volume - expects [batch, channels, samples]"""
        # Handle batch dimension
        if waveform.dim() == 3:
            audio_np = waveform[0, 0].cpu().numpy()  # Get first batch, first channel
        elif waveform.dim() == 2:
            audio_np = waveform[0].cpu().numpy()  # Get first channel
        else:
            audio_np = waveform.cpu().numpy()
        
        peak = np.abs(audio_np).max()
        if peak == 0:
            return waveform
        
        target_peak = 10 ** (target_peak_db / 20)
        gain_factor = target_peak / peak
        
        # Apply gain to original tensor
        waveform = waveform * gain_factor
        waveform = torch.clamp(waveform, -1.0, 1.0)
        
        return waveform
    
    def generate_speech(self, text, model_type, language, exaggeration, cfg_weight, 
                       temperature, speed, normalize_volume, target_peak_db, gain, seed,
                       reference_audio=None):
        """Generate speech from text"""
        
        print(f"📝 Text ({len(text)} chars): '{text[:80]}...'")
        
        model = self.load_model(model_type)
        
        print(f"⚡ Using {model_type.title()} on CPU {'(6x faster than real-time)' if model_type == 'turbo' else ''}")
        
        kwargs = {
            "exaggeration": float(exaggeration),
            "cfg_weight": float(cfg_weight),
            "temperature": float(temperature),
        }
        
        if model_type == "multilingual":
            kwargs["language_id"] = language
        
        # Handle reference audio
        temp_path = None
        if reference_audio is not None:
            ref_waveform = reference_audio["waveform"].cpu()
            ref_sr = reference_audio["sample_rate"]
            
            # Convert [batch, channels, samples] to [samples, channels] for saving
            if ref_waveform.dim() == 3:
                ref_waveform = ref_waveform[0].transpose(0, 1)  # [channels, samples] -> [samples, channels]
            elif ref_waveform.dim() == 2:
                ref_waveform = ref_waveform.transpose(0, 1)
            else:
                ref_waveform = ref_waveform.unsqueeze(1)
            
            ref_np = ref_waveform.numpy()
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_path = tmp.name
                sf.write(temp_path, ref_np, ref_sr)
            
            kwargs["audio_prompt_path"] = temp_path
            print(f"🎵 Using reference audio")
        
        print(f"⚙️  exaggeration={exaggeration}, cfg_weight={cfg_weight}, temperature={temperature}")
        if speed != 1.0:
            print(f"   ℹ️  Speed={speed} (post-processing)")
        if seed != 0:
            torch.manual_seed(seed)
            print(f"   ℹ️  Seed={seed} (PyTorch RNG)")
        
        try:
            print("🎙️  Generating on CPU...")
            
            audio_output = model.generate(text, **kwargs)
            
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except:
                    pass
            
            # Convert to tensor
            if isinstance(audio_output, np.ndarray):
                waveform = torch.from_numpy(audio_output).float()
            elif isinstance(audio_output, torch.Tensor):
                waveform = audio_output.float()
            else:
                raise ValueError(f"Unexpected audio output type: {type(audio_output)}")
            
            # ===== FIX: Ensure ComfyUI format [batch, channels, samples] =====
            if waveform.dim() == 1:
                # [samples] -> [1, 1, samples]
                waveform = waveform.unsqueeze(0).unsqueeze(0)
            elif waveform.dim() == 2:
                # Check if [channels, samples] or [samples, channels]
                if waveform.shape[0] < waveform.shape[1]:
                    # Likely [channels, samples] -> [1, channels, samples]
                    waveform = waveform.unsqueeze(0)
                else:
                    # Likely [samples, channels] -> transpose and add batch
                    waveform = waveform.transpose(0, 1).unsqueeze(0)
            elif waveform.dim() == 3:
                # Already [batch, channels, samples]
                pass
            else:
                raise ValueError(f"Unexpected waveform dimensions: {waveform.shape}")
            
            # Apply speed change (post-processing)
            if speed != 1.0:
                current_length = waveform.shape[2]  # samples dimension
                target_length = int(current_length / speed)
                
                # Interpolate along samples dimension
                waveform = torch.nn.functional.interpolate(
                    waveform, 
                    size=target_length, 
                    mode='linear',
                    align_corners=False
                )
            
            # Apply gain
            if gain != 1.0:
                waveform = waveform * gain
            
            # Normalize volume
            if normalize_volume:
                waveform = self.normalize_audio(waveform, target_peak_db)
            
            sample_rate = 24000
            
            duration = waveform.shape[2] / sample_rate  # samples is last dimension
            print(f"✅ Generated {duration:.2f}s @ {sample_rate}Hz (CPU)")
            print(f"   📊 Output shape: {list(waveform.shape)} [batch, channels, samples]")
            
            return ({"waveform": waveform, "sample_rate": sample_rate},)
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except:
                    pass
            raise RuntimeError(f"Error generating speech: {str(e)}")


class ChatterboxLoadReferenceAudio:
    """Load reference audio for voice cloning"""
    
    @classmethod
    def INPUT_TYPES(cls):
        os.makedirs(CHATTERBOX_AUDIO_DIR, exist_ok=True)
        
        try:
            all_files = os.listdir(CHATTERBOX_AUDIO_DIR)
            audio_files = [f for f in all_files 
                          if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'))]
            
            if not audio_files:
                hint_path = os.path.join(CHATTERBOX_AUDIO_DIR, "PUT_AUDIO_FILES_HERE.txt")
                if not os.path.exists(hint_path):
                    with open(hint_path, 'w') as f:
                        f.write("Place your reference audio files (.wav, .mp3, .flac) here\n")
                        f.write("Then restart ComfyUI or refresh the node.\n")
                
                audio_files = ["No audio files found"]
                
        except Exception as e:
            print(f"⚠️  Error reading audio directory {CHATTERBOX_AUDIO_DIR}: {e}")
            audio_files = ["Error reading directory"]
        
        return {
            "required": {
                "audio_file": (sorted(audio_files),),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "load_audio"
    CATEGORY = "audio/chatterbox"
    
    def load_audio(self, audio_file):
        """Load reference audio file"""
        
        if audio_file in ["No audio files found", "Error reading directory"]:
            raise ValueError(f"No audio files in:\n{CHATTERBOX_AUDIO_DIR}\n\n"
                           f"Place audio files (.wav, .mp3, .flac, .ogg, .m4a) there,\n"
                           f"then restart ComfyUI or refresh the node.")
        
        audio_path = os.path.join(CHATTERBOX_AUDIO_DIR, audio_file)
        
        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")
        
        print(f"📂 Loading: {audio_file}")
        
        try:
            audio_data, sr = sf.read(audio_path, dtype='float32')
            
            if audio_data.ndim == 1:
                # Mono: [samples] -> [1, 1, samples]
                waveform = torch.from_numpy(audio_data).unsqueeze(0).unsqueeze(0)
            else:
                # Stereo: [samples, channels] -> [1, channels, samples]
                waveform = torch.from_numpy(audio_data.T).unsqueeze(0)
            
        except Exception as e:
            print(f"  ⚠️  soundfile failed, trying torchaudio: {e}")
            waveform, sr = torchaudio.load(audio_path)
            
            # torchaudio loads as [channels, samples], add batch dimension
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)  # [1, channels, samples]
        
        duration = waveform.shape[2] / sr
        print(f"  ↳ {duration:.2f}s @ {sr}Hz, shape: {list(waveform.shape)}")
        
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
                "format": (["wav", "flac"],),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save_audio"
    OUTPUT_NODE = True
    CATEGORY = "audio/chatterbox"
    
    def save_audio(self, audio, filename_prefix, format):
        output_dir = folder_paths.get_output_directory()
        
        counter = 1
        while True:
            filename = f"{filename_prefix}_{counter:04d}.{format}"
            filepath = os.path.join(output_dir, filename)
            if not os.path.exists(filepath):
                break
            counter += 1
        
        waveform = audio["waveform"].cpu()
        sample_rate = audio["sample_rate"]
        
        # Convert [batch, channels, samples] -> [samples, channels] for saving
        if waveform.dim() == 3:
            # Take first batch: [batch, channels, samples] -> [channels, samples] -> [samples, channels]
            waveform = waveform[0].transpose(0, 1).numpy()
        elif waveform.dim() == 2:
            # [channels, samples] -> [samples, channels]
            waveform = waveform.transpose(0, 1).numpy()
        else:
            # [samples] -> [samples, 1]
            waveform = waveform.unsqueeze(1).numpy()
        
        sf.write(filepath, waveform, sample_rate, format=format.upper())
        
        print(f"💾 Saved: {filename}")
        
        return ()


NODE_CLASS_MAPPINGS = {
    "ChatterboxTTSNode": ChatterboxTTSNode,
    "ChatterboxLoadReferenceAudio": ChatterboxLoadReferenceAudio,
    "ChatterboxPresets": ChatterboxPresets,
    "ChatterboxSaveAudio": ChatterboxSaveAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatterboxTTSNode": "Chatterbox TTS (CPU)",
    "ChatterboxLoadReferenceAudio": "Load Reference Audio (Chatterbox)",
    "ChatterboxPresets": "Chatterbox Presets",
    "ChatterboxSaveAudio": "Save Audio (Chatterbox)",
}
