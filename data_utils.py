import unicodedata
import pandas as pd
from datasets import Dataset, Audio


def clean_text(text):
    """Normalize and clean transcription text."""
    text = unicodedata.normalize('NFC', text)
    return text.strip()


def load_dataset(csv_path: str, test_size: float = 0.1, seed: int = 42):
    """
    Load Ge'ez speech dataset from CSV and split into train/test.

    CSV format:
        audio_path  : path to .wav file
        text        : Ge'ez transcription

    Returns:
        train_dataset, test_dataset (HuggingFace Dataset objects)
    """
    df = pd.read_csv(csv_path)
    df['path'] = df['audio_path'].str.replace('All_voices', 'voice')
    df['transcription'] = df['text'].apply(clean_text)
    df = df.drop(columns=['audio_path', 'text'])

    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("path", Audio(sampling_rate=16000))

    split = dataset.train_test_split(test_size=test_size, seed=seed)
    return split['train'], split['test']
