"""
evaluate.py — Evaluate fine-tuned Whisper model on Ge'ez test set using WER.

Usage:
    python evaluate.py --model_dir ./whisper-finetuned --data_path data/geez_data.csv
"""

import argparse
import torch
from tqdm import tqdm
import evaluate as hf_evaluate
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from data_utils import load_dataset


def evaluate_model(model_dir: str, data_path: str):

    # Load fine-tuned model and processor
    print(f"Loading model from {model_dir} ...")
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    processor = WhisperProcessor.from_pretrained(model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Load test set
    _, test_dataset = load_dataset(data_path)

    wer_metric = hf_evaluate.load("wer")
    predictions = []
    references = []

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="am", task="transcribe"
    )

    print("Running inference on test set...")
    for sample in tqdm(test_dataset):
        audio_data = sample["path"]
        input_features = processor(
            audio_data["array"], sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
            )

        predicted_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        reference_text = sample["transcription"].strip()

        predictions.append(predicted_text)
        references.append(reference_text)

    wer = wer_metric.compute(predictions=predictions, references=references)
    print(f"\n📊 Full WER: {wer:.4f}  ({wer * 100:.2f}%)")
    return wer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Whisper Ge'ez ASR model")
    parser.add_argument("--model_dir", type=str, default="./whisper-finetuned",
                        help="Path to fine-tuned model directory")
    parser.add_argument("--data_path", type=str, default="data/geez_data.csv",
                        help="Path to CSV dataset file")
    args = parser.parse_args()
    evaluate_model(args.model_dir, args.data_path)
