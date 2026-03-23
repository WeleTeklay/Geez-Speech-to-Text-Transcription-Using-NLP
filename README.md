# 🎙️ Whisper Fine-Tuning for Ge'ez Speech-to-Text Transcription

> Fine-tuning OpenAI's Whisper model for automatic speech recognition (ASR) on **Ge'ez** — an ancient Semitic script used in Ethiopian and Eritrean languages — addressing the challenge of ASR for low-resource languages with non-Latin scripts.

---

## 📌 Overview

Ge'ez (ግዕዝ) is a classical Ethiopic script used in liturgical and written forms of languages such as Amharic and Tigrigna. Despite its cultural significance, it remains severely underrepresented in modern ASR systems. This project fine-tunes [OpenAI's Whisper-Small](https://huggingface.co/openai/whisper-small) on a custom Ge'ez audio dataset to build a baseline transcription system for this low-resource language.

---

## ✨ Key Features

- Fine-tuned `openai/whisper-small` on a custom Ge'ez speech dataset
- Custom Ge'ez Unicode token extension (`U+1200–U+137F`) for accurate script handling
- Full training pipeline: preprocessing → fine-tuning → evaluation
- WER-based evaluation using HuggingFace `evaluate`
- Early stopping and best-model checkpointing for efficient training
- Ready-to-use inference pipeline for new audio samples

---

## 🗂️ Project Structure

```
whisper-geez-asr/
│
├── data/
│   └── geez_data.csv          # Dataset: audio paths + transcriptions
│
├── whisper_geez_finetune.py   # Main training script
├── evaluate.py                # WER evaluation on test set
├── inference.py               # Run transcription on new audio
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Library |
|---|---|
| Model | `openai/whisper-small` via HuggingFace Transformers |
| Training | `Seq2SeqTrainer` with early stopping |
| Audio Processing | `torchaudio`, `librosa` |
| Tokenization | Extended Whisper tokenizer with Ge'ez Unicode block |
| Evaluation | `evaluate` (WER metric) |
| Data | HuggingFace `datasets`, custom CSV |
| Environment | Google Colab (GPU), Python 3.10+ |

---

## ⚙️ Setup

```bash
git clone https://github.com/<your-username>/whisper-geez-asr.git
cd whisper-geez-asr
pip install -r requirements.txt
sudo apt-get install -y ffmpeg
```

**requirements.txt**
```
torch
torchaudio
transformers
datasets
jiwer
evaluate
accelerate
librosa
ethiopic
```

---

## 📦 Dataset Format

The dataset is a CSV file with the following columns:

```csv
audio_path,text
All_voices/sample_001.wav,ሰላም ዓለም
All_voices/sample_002.wav,እንደምን አለህ
...
```

Audio files should be in `.wav` format, mono channel, resampled to **16kHz**.

---

## 🚀 Training

```python
# Key training configuration
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-finetuned",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    learning_rate=1e-5,
    eval_strategy="steps",
    eval_steps=100,
    metric_for_best_model="wer",
    greater_is_better=False,       # Lower WER = better
    load_best_model_at_end=True,
    gradient_checkpointing=True,
    fp16=True,
)
```

The tokenizer is extended with Ge'ez Unicode characters (`U+1200–U+137F`) before training to ensure proper script coverage:

```python
geez_tokens = [chr(c) for c in range(0x1200, 0x1380)
               if unicodedata.category(chr(c)) in ['Lo', 'Mn']]
tokenizer.add_tokens(geez_tokens)
model.resize_token_embeddings(len(tokenizer))
```

---

## 📊 Evaluation

Evaluation uses **Word Error Rate (WER)** on a held-out test set (10% split, `seed=42`):

```bash
python evaluate.py
```

```
Full WER: 0.2000  (20.00%)
```

> WER measures the percentage of words incorrectly transcribed. Lower is better.

---

## 🔍 Inference

Run transcription on a new audio file:

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
import torchaudio

model = WhisperForConditionalGeneration.from_pretrained("./whisper-finetuned").to("cuda")
processor = WhisperProcessor.from_pretrained("./whisper-finetuned")

asr = pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=0
)

waveform, sr = torchaudio.load("your_audio.wav")
if sr != 16000:
    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

result = asr(waveform.squeeze().numpy(), generate_kwargs={"language": "<|am|>"})
print("Transcription:", result["text"])
```

---

## 🌍 Why This Matters

Ge'ez and related Ethiopic scripts serve over **120 million speakers** across Ethiopia and Eritrea. Despite this, they remain among the most underrepresented language families in speech technology research. This project is a step toward:

- Building accessible voice interfaces for Ethiopic-script languages
- Establishing ASR baselines for low-resource Semitic languages
- Supporting digital preservation of classical Ge'ez texts

---

## 👤 Author

**Weldesemayat Teklay Gebre**
BSc. Computer Science and Engineering — Mekelle Institute of Technology, Mekelle University
📧 weleteklay2127@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/weldesemayat-teklay-66000019a)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) for the base model
- [HuggingFace](https://huggingface.co) for the Transformers and Datasets libraries
- Mekelle Institute of Technology for academic support
