# chatterbox_nodes.py - Complete Chatterbox TTS for ComfyUI (ALL 3 MODELS)
import torch
import torchaudio
import soundfile as sf
import numpy as np
import os
import folder_paths
import tempfile
import random

# 🔧 CPU COMPATIBILITY PATCH (from your working code)
print("🔧 Applying Chatterbox CPU compatibility patches...")

# Patch 1: Attention implementation fix
try:
    import chatterbox.models.t3.inference.alignment_stream_analyzer as asa
    original_add_attention_spy = asa.AlignmentStreamAnalyzer._add_attention_spy
    
    def patched_add_attention_spy(self, tfmr, i, layer_idx, head_idx):
        tfmr.config.attn_implementation = "eager"
        original_add_attention_spy(self, tfmr, i, layer_idx, head_idx)
    
    asa.AlignmentStreamAnalyzer._add_attention_spy = patched_add_attention_spy
    print("✅ Attention patch applied")
except Exception as e:
    print(f"⚠️  Attention patch skipped: {e}")

# Patch 2: Force CPU loading
_original_torch_load = torch.load

def torch_load_cpu_patch(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = torch.device("cpu")
    return _original_torch_load(*args, **kwargs)

torch.load = torch_load_cpu_patch
print("✅ CPU loading patch applied")


class ChatterboxTTSNode:
    """
    Universal Chatterbox TTS Node
    Supports ALL 3 models: English, Multilingual (23 langs), Turbo (fast)
    CPU-optimized with compatibility patches
    """
    
    MODEL_TYPES = {
        "english": "Chatterbox (English - 500M)",
        "turbo": "Chatterbox Turbo (English - 350M - 6x Faster)",
        "multilingual": "Chatterbox Multilingual (23 Languages - 500M)",
    }
    
    def __init__(self):
        self.model = None
        self.current_model_type = None
        self.device = "cpu"  # Force CPU
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hi there! This is Chatterbox, a state of the art open source text to speech system."
                }),
                "model_type": (list(cls.MODEL_TYPES.keys()), {
                    "default": "turbo",
                    "tooltip": "Turbo = fastest, Multilingual = 23 languages"
                }),
            },
            "optional": {
                "language": (["en", "ar", "da", "de", "el", "es", "fi", "fr", "he", 
                             "hi", "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", 
                             "ru", "sv", "sw", "tr", "zh"], {
                    "default": "en",
                    "tooltip": "Only for multilingual model"
                }),
                "reference_audio": ("AUDIO",),
                
                # Chatterbox-specific controls
                "exaggeration": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.25,
                    "max": 2.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Emotion/expressiveness (0.5=neutral)"
                }),
                "cfg_weight": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "CFG/Pace (0=no accent transfer for language switching)"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.05,
                    "max": 5.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Creativity/randomness"
                }),
                
                # Post-processing
                "speed": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.05,
                    "display": "slider",
                }),
                "normalize_volume": ("BOOLEAN", {
                    "default": True,
                }),
                "target_peak_db": ("FLOAT", {
                    "default": -3.0,
                    "min": -20.0,
                    "max": 0.0,
                    "step": 0.5,
                    "display": "slider",
                }),
                "gain": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.1,
                    "display": "slider",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffff,
                }),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "audio/generation"
    OUTPUT_NODE = False
    
    def load_model(self, model_type):
        """Load the appropriate Chatterbox model"""
        if self.model is None or self.current_model_type != model_type:
            try:
                print(f"🎙️ Loading {self.MODEL_TYPES[model_type]}...")
                
                if model_type == "english":
                    from chatterbox.tts import ChatterboxTTS
                    self.model = ChatterboxTTS.from_pretrained(device=self.device)
                    print("✅ Chatterbox English (500M) loaded")
                    
                elif model_type == "turbo":
                    from chatterbox.tts_turbo import ChatterboxTurboTTS
                    self.model = ChatterboxTurboTTS.from_pretrained(device=self.device)
                    print("✅ Chatterbox Turbo (350M - 6x faster) loaded")
                    print("   💡 Supports paralinguistic tags: [laugh], [chuckle], [sigh], [gasp], [cough]")
                    
                elif model_type == "multilingual":
                    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                    self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
                    print("✅ Chatterbox Multilingual (500M - 23 languages) loaded")
                
                self.current_model_type = model_type
                
                # Ensure CPU device
                if hasattr(self.model, 'to'):
                    self.model.to(self.device)
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise RuntimeError(
                    f"Failed to load Chatterbox: {str(e)}\n\n"
                    "Installation:\n"
                    "  pip install chatterbox-tts\n"
                    "Or:\n"
                    "  pip install git+https://github.com/resemble-ai/chatterbox.git\n\n"
                    "If transformers error:\n"
                    "  pip install transformers==4.46.3 --force-reinstall"
                )
        
        return self.model
    
    def set_seed(self, seed):
        """Set random seed"""
        if seed > 0:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
    
    def check_paralinguistic_tags(self, text, model_type):
        """Check and warn about paralinguistic tags"""
        tags = ["[laugh]", "[chuckle]", "[sigh]", "[gasp]", "[cough]", "[pause]"]
        found_tags = [tag for tag in tags if tag.lower() in text.lower()]
        
        if found_tags:
            if model_type == "turbo":
                print(f"🎭 Paralinguistic tags detected: {', '.join(found_tags)}")
            else:
                print(f"⚠️  Tags detected but NOT supported in {model_type} model: {', '.join(found_tags)}")
                print("   💡 Use 'turbo' model for paralinguistic tag support")
    
    def generate_speech(self, text, model_type,
                       language="en",
                       reference_audio=None,
                       exaggeration=0.5, cfg_weight=0.5, temperature=0.8,
                       speed=1.0,
                       normalize_volume=True, target_peak_db=-3.0, gain=1.5,
                       seed=0):
        """Generate speech with Chatterbox TTS"""
        
        # Set seed
        self.set_seed(seed)
        if seed > 0:
            print(f"🎲 Seed: {seed}")
        
        # Load model
        model = self.load_model(model_type)
        
        # Truncate text to 300 chars
        text = text[:300].strip()
        
        if not text:
            raise ValueError("Text cannot be empty")
        
        # Check for paralinguistic tags
        self.check_paralinguistic_tags(text, model_type)
        
        print(f"📝 Text ({len(text)} chars): '{text[:80]}{'...' if len(text) > 80 else ''}'")
        
        if model_type == "multilingual":
            print(f"🌍 Language: {language}")
        elif model_type == "turbo":
            print("⚡ Using Turbo (6x faster than real-time)")
        
        temp_audio_path = None
        
        try:
            # Prepare kwargs
            kwargs = {
                "exaggeration": exaggeration,
                "temperature": temperature,
                "cfg_weight": cfg_weight,
            }
            
            # Handle reference audio
            if reference_audio is not None:
                print("🎵 Processing reference audio...")
                ref_waveform = reference_audio['waveform']
                ref_sample_rate = reference_audio['sample_rate']
                
                if isinstance(ref_waveform, torch.Tensor):
                    ref_waveform = ref_waveform.cpu().numpy()
                
                if ref_waveform.ndim == 3:
                    ref_waveform = ref_waveform[0]
                
                if ref_waveform.shape[0] > 1:
                    ref_waveform = ref_waveform.mean(axis=0, keepdims=True)
                    print("  ↳ Converted to mono")
                
                if ref_sample_rate != 24000:
                    print(f"  ↳ Resampling {ref_sample_rate}Hz → 24000Hz")
                    ref_waveform_tensor = torch.from_numpy(ref_waveform).float()
                    ref_waveform_tensor = torchaudio.functional.resample(
                        ref_waveform_tensor, ref_sample_rate, 24000
                    )
                    ref_waveform = ref_waveform_tensor.numpy()
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    temp_audio_path = tmp.name
                    sf.write(temp_audio_path, ref_waveform.T, 24000)
                
                kwargs["audio_prompt_path"] = temp_audio_path
            
            print(f"⚙️  exag={exaggeration}, cfg={cfg_weight}, temp={temperature}")
            print("🎙️  Generating...")
            
            # Generate (API differs for multilingual)
            if model_type == "multilingual":
                audio_output = model.generate(text, language_id=language, **kwargs)
            else:
                audio_output = model.generate(text, **kwargs)
            
            print("✅ Generation complete!")
            
            # Convert to tensor
            if isinstance(audio_output, np.ndarray):
                audio_tensor = torch.from_numpy(audio_output).float()
            else:
                audio_tensor = audio_output.float()
            
            # Ensure shape
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
            elif audio_tensor.dim() == 2:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            duration = audio_tensor.shape[-1] / model.sr
            print(f"⏱️  Duration: {duration:.2f}s")
            
            # Speed adjustment
            if speed != 1.0:
                print(f"⚡ Speed: {speed}x")
                target_sr = int(model.sr * speed)
                audio_tensor = torchaudio.functional.resample(
                    audio_tensor, orig_freq=target_sr, new_freq=model.sr
                )
            
            # Gain
            if gain != 1.0:
                audio_tensor = audio_tensor * gain
                print(f"🔊 Gain: {gain}x")
            
            # Normalize
            if normalize_volume:
                target_peak = 10 ** (target_peak_db / 20.0)
                current_peak = audio_tensor.abs().max()
                
                if current_peak > 0:
                    scale_factor = target_peak / current_peak
                    audio_tensor = audio_tensor * scale_factor
                    final_peak_db = 20 * torch.log10(audio_tensor.abs().max())
                    print(f"🔊 Normalized: {final_peak_db.item():.1f} dB")
            else:
                max_val = audio_tensor.abs().max()
                if max_val > 1.0:
                    audio_tensor = audio_tensor / max_val
            
            return ({
                "waveform": audio_tensor,
                "sample_rate": model.sr
            },)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Error generating speech: {str(e)}")
        
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                    print("🧹 Cleaned up")
                except:
                    pass


class ChatterboxPresets:
    """Preset configurations"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": ([
                    "balanced",
                    "expressive", 
                    "dramatic",
                    "consistent",
                    "fast_paced",
                    "slow_deliberate",
                    "audiobook",
                    "language_transfer",
                ], {
                    "default": "balanced"
                }),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("exaggeration", "cfg_weight", "temperature", "target_peak_db", "gain")
    FUNCTION = "get_preset"
    CATEGORY = "audio/generation/presets"
    
    def get_preset(self, preset):
        presets = {
            # exag, cfg, temp, peak_db, gain
            "balanced":          (0.5,  0.5,  0.8,  -3.0, 1.5),
            "expressive":        (0.7,  0.3,  1.0,  -3.0, 1.6),
            "dramatic":          (0.9,  0.3,  1.2,  -3.0, 1.7),
            "consistent":        (0.3,  0.7,  0.5,  -3.0, 1.5),
            "fast_paced":        (0.6,  0.3,  0.8,  -3.0, 1.5),
            "slow_deliberate":   (0.4,  0.7,  0.6,  -3.0, 1.5),
            "audiobook":         (0.4,  0.6,  0.6,  -6.0, 1.4),
            "language_transfer": (0.5,  0.0,  0.8,  -3.0, 1.5),
        }
        
        values = presets.get(preset, presets["balanced"])
        print(f"🎚️  Preset: {preset}")
        return values


class ChatterboxLoadReferenceAudio:
    """Load reference audio"""
    
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = []
        if os.path.exists(input_dir):
            files = [f for f in os.listdir(input_dir)
                    if f.endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a'))]
        
        return {
            "required": {
                "audio_file": (sorted(files) if files else ["No audio files found"],),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("reference_audio",)
    FUNCTION = "load_audio"
    CATEGORY = "audio/loading"
    
    def load_audio(self, audio_file):
        input_dir = folder_paths.get_input_directory()
        audio_path = os.path.join(input_dir, audio_file)
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            print(f"📂 Loading: {audio_file}")
            
            data, sample_rate = sf.read(audio_path, dtype='float32')
            waveform = torch.from_numpy(data).float()
            
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.T
            
            waveform = waveform.unsqueeze(0)
            
            duration = waveform.shape[-1] / sample_rate
            print(f"  ↳ {duration:.2f}s @ {sample_rate}Hz")
            
            return ({
                "waveform": waveform,
                "sample_rate": sample_rate
            },)
            
        except Exception as e:
            raise RuntimeError(f"Error loading audio: {str(e)}")


class ChatterboxSaveAudio:
    """Save audio"""
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "chatterbox_audio"}),
            },
            "optional": {
                "format": (["wav", "flac", "ogg"],),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "save_audio"
    OUTPUT_NODE = True
    CATEGORY = "audio/output"
    
    def save_audio(self, audio, filename_prefix="chatterbox_audio", format="wav"):
        waveform = audio['waveform']
        sample_rate = audio['sample_rate']
        
        if waveform.dim() == 3:
            waveform = waveform[0]
        
        audio_numpy = waveform.cpu().numpy().T
        
        counter = 0
        while True:
            filename = f"{filename_prefix}_{counter:05d}.{format}"
            filepath = os.path.join(self.output_dir, filename)
            if not os.path.exists(filepath):
                break
            counter += 1
        
        try:
            sf.write(filepath, audio_numpy, sample_rate)
            duration = len(audio_numpy) / sample_rate
            print(f"💾 Saved: {filepath}")
            print(f"  ↳ {duration:.2f}s")
            
            return {"ui": {"audio": [filename]}}
            
        except Exception as e:
            raise RuntimeError(f"Error saving: {str(e)}")


NODE_CLASS_MAPPINGS = {
    "ChatterboxTTSNode": ChatterboxTTSNode,
    "ChatterboxPresets": ChatterboxPresets,
    "ChatterboxLoadReferenceAudio": ChatterboxLoadReferenceAudio,
    "ChatterboxSaveAudio": ChatterboxSaveAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatterboxTTSNode": "🌍 Chatterbox TTS (All Models)",
    "ChatterboxPresets": "🎚️ Chatterbox Presets",
    "ChatterboxLoadReferenceAudio": "📂 Chatterbox Load Audio",
    "ChatterboxSaveAudio": "💾 Chatterbox Save Audio",
}
