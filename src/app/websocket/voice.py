"""Voice resolution and mapping logic."""

import os


class VoiceResolver:
    """Handles voice resolution and mapping for the WebSocket API."""
    
    def __init__(self):
        # Voice mapping for API
        self.voice_mapping = {
            "female": os.getenv("DEFAULT_VOICE_FEMALE", "af_heart"),
            "male": os.getenv("DEFAULT_VOICE_MALE", "am_michael"),
        }
        self.default_voice = "female"
    
    def resolve_voice(self, requested: str) -> str:
        """Resolve API voice name to Kokoro voice id. Only 'female' and 'male' are allowed."""
        if not requested:
            return self.voice_mapping["female"]
        req_lower = requested.lower()
        if req_lower in ("female", "f"):
            return self.voice_mapping["female"]
        if req_lower in ("male", "m"):
            return self.voice_mapping["male"]
        raise ValueError(f"Voice '{requested}' not found (allowed: female, male)")
    
    def get_default_voice(self) -> str:
        """Get the default voice name."""
        return self.default_voice
    
    def set_default_voice(self, voice: str) -> None:
        """Set the default voice name (must be valid)."""
        # Validate the voice first
        self.resolve_voice(voice)
        self.default_voice = voice
