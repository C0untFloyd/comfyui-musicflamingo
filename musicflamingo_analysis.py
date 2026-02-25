import os
import tempfile
from typing import Dict, Any, Tuple

import torch
import torchaudio
from transformers import AutoProcessor
from transformers.models.audioflamingo3.modeling_audioflamingo3 import (
    AudioFlamingo3ForConditionalGeneration,
)


MODEL_ID = "nvidia/music-flamingo-hf"

_processor = None
_model = None


def _get_music_flamingo() -> Tuple[AutoProcessor, AudioFlamingo3ForConditionalGeneration]:
    """
    Lazily load the Music Flamingo processor + model once per process.
    """
    global _processor, _model

    if _processor is None or _model is None:
        _processor = AutoProcessor.from_pretrained(MODEL_ID)

        # Prefer bfloat16 on supported GPUs, otherwise fall back to fp16 (GPU) or fp32 (CPU).
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                preferred_dtype = torch.bfloat16
            else:
                preferred_dtype = torch.float16
        else:
            preferred_dtype = torch.float32

        try:
            _model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
                MODEL_ID,
                device_map="auto",
                torch_dtype=preferred_dtype,
            )
        except (TypeError, RuntimeError):
            # If the chosen dtype is not supported on this device, fall back to fp32.
            _model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
                MODEL_ID,
                device_map="auto",
                torch_dtype=torch.float32,
            )

    return _processor, _model


class MusicFlamingoAnalysis:
    """
    ComfyUI node: analyze an audio clip with Music Flamingo and return a text description.
    Expects audio from a regular Load Audio node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "Describe this track in full detail - tell me the genre, tempo, and key, "
                            "then dive into the instruments and describe the song structure."
                        ),
                    },
                ),
                "max_new_tokens": (
                    "INT",
                    {
                        "default": 512,
                        "min": 1,
                        "max": 1024,
                        "step": 8,
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "analyze"
    CATEGORY = "audio/MusicFlamingo"

    def analyze(self, audio: Dict[str, Any], prompt: str, max_new_tokens: int) -> Tuple[str]:
        """
        `audio` is a Comfy AUDIO dict: {"waveform": [1, C, T] float32, "sample_rate": int}.
        """
        if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
            raise ValueError("MusicFlamingoAnalysis expects an AUDIO dict with 'waveform' and 'sample_rate'.")

        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])

        if not isinstance(waveform, torch.Tensor):
            raise ValueError("MusicFlamingoAnalysis expects 'waveform' to be a torch.Tensor.")

        # Expect shape [1, C, T], restrict to batch size 1 for now.
        if waveform.dim() != 3 or waveform.shape[0] != 1:
            raise ValueError(
                f"MusicFlamingoAnalysis expects audio with shape [1, C, T]; got {tuple(waveform.shape)}."
            )

        # Convert to [C, T] for torchaudio.save
        waveform = waveform.squeeze(0).cpu()

        processor, model = _get_music_flamingo()

        # Save the incoming audio tensor to a temporary WAV file and pass its path to the processor,
        # matching the reference Music Flamingo example that uses an audio file path.
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "input.wav")
            torchaudio.save(audio_path, waveform, sample_rate)

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "audio", "path": audio_path},
                    ],
                }
            ]

            inputs = processor.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            ).to(model.device)

            # Cast float32 tensors to the model's actual dtype (bfloat16/fp16/fp32).
            try:
                model_dtype = next(p.dtype for p in model.parameters() if p is not None)
            except StopIteration:
                model_dtype = torch.float32

            inputs = {
                k: v.to(model_dtype) if isinstance(v, torch.Tensor) and v.dtype == torch.float32 else v
                for k, v in inputs.items()
            }

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

            # Slice off the input tokens and decode only the generated continuation.
            generated_only = outputs[:, inputs["input_ids"].shape[1] :]
            decoded_outputs = processor.batch_decode(generated_only, skip_special_tokens=True)

        description = ""
        if isinstance(decoded_outputs, list) and decoded_outputs:
            description = decoded_outputs[0]
        elif isinstance(decoded_outputs, str):
            description = decoded_outputs

        return (description,)


NODE_CLASS_MAPPINGS = {
    "MusicFlamingoAnalysis": MusicFlamingoAnalysis,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MusicFlamingoAnalysis": "Music Flamingo Analysis",
}

