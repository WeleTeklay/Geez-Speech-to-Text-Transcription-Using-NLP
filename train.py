"""
train.py — Fine-tune OpenAI Whisper-Small on Ge'ez speech data.

Usage:
    python train.py --data_path geez_data.csv --output_dir ./whisper-finetuned
"""

import argparse
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
import evaluate
from transformers import (
    WhisperProcessor,
    WhisperTokenizer,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)

from data_utils import clean_text, load_dataset


# ─── Config ───────────────────────────────────────────────────────────────────

MODEL_NAME = "openai/whisper-small"
TARGET_LANGUAGE = "am"   # 'am' for Amharic/Ge'ez


# ─── Data Collator ────────────────────────────────────────────────────────────

@dataclass
class DataCollator:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        batch["labels"] = labels["input_ids"].masked_fill(
            labels["attention_mask"].ne(1), -100
        )
        return batch


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(data_path: str, output_dir: str):

    # Load processor and tokenizer
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
    tokenizer = WhisperTokenizer.from_pretrained(
        MODEL_NAME, language=TARGET_LANGUAGE, task="transcribe"
    )
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME, language=TARGET_LANGUAGE, task="transcribe"
    )

    # Extend tokenizer with Ge'ez Unicode block (U+1200–U+137F)
    geez_tokens = [
        chr(c) for c in range(0x1200, 0x1380)
        if unicodedata.category(chr(c)) in ['Lo', 'Mn']
    ]
    tokenizer.add_tokens(geez_tokens)
    processor.tokenizer = tokenizer

    # Load model and resize embeddings for new tokens
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Load and preprocess dataset
    train_dataset, test_dataset = load_dataset(data_path)

    def preprocess(batch):
        audio = batch["path"]
        batch["input_features"] = feature_extractor(
            audio["array"], sampling_rate=16000
        ).input_features[0]
        batch["labels"] = tokenizer(clean_text(batch["transcription"])).input_ids
        return batch

    train_dataset = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(preprocess, remove_columns=test_dataset.column_names)

    # WER metric
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": wer_metric.compute(predictions=pred_str, references=label_str)}

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=300,
        logging_steps=50,
        predict_with_generate=True,
        generation_max_length=225,
        learning_rate=1e-5,
        num_train_epochs=10,
        gradient_checkpointing=True,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=["tensorboard"],
    )

    data_collator = DataCollator(processor)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    # Save final model and processor
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"\n✅ Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on Ge'ez speech data")
    parser.add_argument("--data_path", type=str, default="data/geez_data.csv",
                        help="Path to CSV dataset file")
    parser.add_argument("--output_dir", type=str, default="./whisper-finetuned",
                        help="Directory to save the fine-tuned model")
    args = parser.parse_args()
    main(args.data_path, args.output_dir)
