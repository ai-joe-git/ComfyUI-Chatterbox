"""
ComfyUI-Chatterbox
All 3 Chatterbox TTS models for ComfyUI
- Chatterbox (English 500M)
- Chatterbox Turbo (English 350M - 6x faster)
- Chatterbox Multilingual (23 languages 500M)
"""

from .chatterbox_nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS
)

# Apply XPU dtype fix
try:
    from .xpu_dtype_fix import apply_xpu_dtype_patch
    apply_xpu_dtype_patch()
except Exception as e:
    print(f"Note: XPU dtype patch not applied: {e}")


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
