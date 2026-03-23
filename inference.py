"""
inference.py — Run Ge'ez speech transcription on a single audio file.

Usage:
    python inference.py --model_dir ./whisper-finetuned --audio_path your_audio.wav
"""

import argparse
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline


def transcribe(model_dir: str, audio_path: str) -> str:

    # Load model and processor
    print(f"Loading model from {model_dir} ...")
    model = WhisperForConditionalGeneration.from_pretrained(model_dir, local_files_only=True)
    processor = WhisperProcessor.from_pretrained(model_dir, local_files_only=True)

    device = 0 if torch.cuda.is_available() else -1

    asr = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
    )

    # Load and resample audio to 16kHz
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)

    audio_array = waveform.squeeze().numpy()

    # Transcribe
    result = asr(audio_array, generate_kwargs={"language": "<|am|>"})
    transcription = result["text"]

    print(f"\n🎙️  Audio   : {audio_path}")
    print(f"📝 Transcription: {transcription}")
    return transcription


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ge'ez speech transcription inference")
    parser.add_argument("--model_dir", type=str, default="./whisper-finetuned",
                        help="Path to fine-tuned model directory")
    parser.add_argument("--audio_path", type=str, required=True,
                        help="Path to input .wav audio file")
    args = parser.parse_args()
    transcribe(args.model_dir, args.audio_path)
