#  Whisper Fine-Tuning for Ge'ez Speech-to-Text Transcription

> Fine-tuning OpenAI's Whisper model for automatic speech recognition (ASR) on **Ge'ez** — an ancient Semitic script used in Ethiopian and Eritrean languages — addressing the challenge of ASR for low-resource languages with non-Latin scripts.

---

##  Overview

Ge'ez (ግዕዝ) is a classical Ethiopic script used in liturgical and written forms of languages such as Amharic and Tigrigna. Despite its cultural significance, it remains severely underrepresented in modern ASR systems. This project fine-tunes [OpenAI's Whisper-Small](https://huggingface.co/openai/whisper-small) on a custom Ge'ez audio dataset to build a baseline transcription system for this low-resource language.

---

##  Key Features

- Fine-tuned `openai/whisper-small` on a custom Ge'ez speech dataset
- Custom Ge'ez Unicode token extension (`U+1200–U+137F`) for accurate script handling
- Full training pipeline: preprocessing → fine-tuning → evaluation
- WER-based evaluation using HuggingFace `evaluate`
- Early stopping and best-model checkpointing for efficient training
- Ready-to-use inference pipeline for new audio samples

---

##  Project Structure

```
Geez-Speech-to-Text-Transcription-Using-NLP/
│
├── train.py          # Fine-tuning pipeline
├── evaluate.py       # WER evaluation on test set
├── inference.py      # Transcribe a single audio file
├── data_utils.py     # Dataset loading and preprocessing
├── requirements.txt  # Dependencies
├── LICENSE
└── README.md
```

---

##  Tech Stack

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

##  Setup

```bash
git clone https://github.com/WeleTeklay/Geez-Speech-to-Text-Transcription-Using-NLP.git
cd Geez-Speech-to-Text-Transcription-Using-NLP
pip install evaluate --force-reinstall
pip install -r requirements.txt
sudo apt-get install -y ffmpeg
```

> **Note:** `evaluate` must be installed before other packages to avoid import conflicts with `transformers 5.x`.

---

##  Dataset Format

The dataset is a CSV file with two columns:

```
audio_path,text
All_voices/sample_001.wav,ሰላም ዓለም
All_voices/sample_002.wav,እንደምን አለህ
```

Audio files should be `.wav` format, mono channel, resampled to **16kHz**.

> The dataset is not included in this repository due to privacy considerations.

---

##  Training

```bash
python train.py \
  --data_path path/to/geez_data.csv \
  --output_dir ./whisper-finetuned
```

The tokenizer is extended with Ge'ez Unicode characters (`U+1200–U+137F`) before training to ensure proper script coverage for the Ethiopic writing system.

---

##  Evaluation

```bash
python evaluate.py \
  --model_dir ./whisper-finetuned \
  --data_path path/to/geez_data.csv
```

Evaluation is performed on a held-out test set (10% split, `seed=42`) using **Word Error Rate (WER)**:

```
 Full WER: 0.2000  (20.00%)
```

> A 20% WER is competitive for low-resource languages with non-Latin scripts, where baseline multilingual models without fine-tuning typically perform significantly worse.

---

##  Inference

```bash
python inference.py \
  --model_dir ./whisper-finetuned \
  --audio_path your_audio.wav
```

---

##  Why This Matters

Ge'ez is a classical Semitic language used primarily in the Ethiopian Orthodox and Eritrean Orthodox churches as a liturgical language. Despite its deep cultural and religious significance across Ethiopia and Eritrea, it remains almost entirely absent from modern speech technology research. This project is a step toward:

- Preserving and digitizing classical Ge'ez liturgical texts through speech technology
- Establishing ASR baselines for an extremely low-resource and underrepresented language
- Supporting the broader goal of inclusive NLP for African and Semitic languages

---

##  Author

**Weldesemayat Teklay Gebre**
BSc. Computer Science and Engineering — Mekelle Institute of Technology, Mekelle University

📧 weleteklay2127@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/weldesemayat-teklay-66000019a)

---



##  Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) for the base model
- [HuggingFace](https://huggingface.co) for the Transformers and Datasets libraries
- Mekelle Institute of Technology for academic support
