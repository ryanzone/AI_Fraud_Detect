import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import os
import random
import io
import numpy as np

# ============================================================
# AudioDataset - Walks UK/ and USA/ directories
# Labels by filename: original.* = real(1), synthetic_*.* = fake(0)
# ============================================================

SUPPORTED_EXTENSIONS = ('.wav', '.mp3', '.flac', '.m4a')

# --------------- Audio Loading Helper ---------------
# torchaudio cannot load .m4a on Windows without ffmpeg.
# We use subprocess + imageio-ffmpeg (bundled ffmpeg binary) as a fallback.

def _get_ffmpeg_path():
    """Get bundled ffmpeg path from imageio-ffmpeg."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None

def load_audio(filepath, target_sr=16000):
    """
    Load any audio file via ffmpeg subprocess (bundled via imageio-ffmpeg).
    Uses Python's built-in wave module to parse the output, avoiding torchaudio codec issues.
    Returns: (waveform [1, T], sample_rate)
    """
    import subprocess
    import wave
    ffmpeg_path = _get_ffmpeg_path()
    if not ffmpeg_path:
        raise RuntimeError("imageio-ffmpeg not installed. Run: pip install imageio-ffmpeg")

    # Use ffmpeg to convert any format to raw PCM wav piped to stdout
    cmd = [
        ffmpeg_path,
        '-i', filepath,
        '-f', 'wav',             # Output format: wav
        '-acodec', 'pcm_s16le',  # 16-bit signed PCM
        '-ac', '1',              # Mono
        '-ar', str(target_sr),   # Target sample rate
        '-v', 'quiet',           # Suppress ffmpeg output
        'pipe:1'                 # Pipe to stdout
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to load {filepath}: {result.stderr.decode()}")

    # Parse the WAV bytes with Python's built-in wave module
    wav_buffer = io.BytesIO(result.stdout)
    with wave.open(wav_buffer, 'rb') as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        audio_bytes = wf.readframes(n_frames)

    # Convert raw bytes to float32 tensor
    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    samples = samples / 32768.0  # Normalize int16 to [-1, 1]
    waveform = torch.from_numpy(samples).unsqueeze(0)  # [1, T]
    return waveform, sr


class AudioDataset(Dataset):
    def __init__(self, filepaths, labels, sample_rate=16000, fixed_length=3*16000):
        """
        filepaths: list of absolute paths to audio files
        labels: list of int labels (0=fake, 1=real)
        """
        self.filepaths = filepaths
        self.labels = labels
        self.sample_rate = sample_rate
        self.fixed_length = fixed_length

        # Spectrogram transform
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        label = self.labels[idx]

        # Load audio (handles .m4a via pydub fallback)
        waveform, sr = load_audio(filepath, self.sample_rate)

        # Resample if necessary
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert stereo to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Pad or truncate to fixed length
        if waveform.shape[1] > self.fixed_length:
            waveform = waveform[:, :self.fixed_length]
        elif waveform.shape[1] < self.fixed_length:
            pad_amount = self.fixed_length - waveform.shape[1]
            waveform = nn.functional.pad(waveform, (0, pad_amount))

        # Extract mel spectrogram
        mel_spec = self.mel_spectrogram(waveform)

        # Convert to log scale (dB)
        log_mel_spec = T.AmplitudeToDB()(mel_spec)

        # Normalize
        log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-6)

        return log_mel_spec, label


def collect_audio_files(data_dirs):
    """
    Walk through UK/ and USA/ style directories and collect audio files with labels.
    Directory structure: <region>/<gender>/<speaker_id>/original.* | synthetic_*.* 
    
    Returns:
        filepaths: list of file paths
        labels: list of labels (0=fake, 1=real)
        speaker_ids: list of unique speaker identifiers (e.g. 'UK_female_1')
    """
    filepaths = []
    labels = []
    speaker_ids = []

    for data_dir in data_dirs:
        region = os.path.basename(data_dir)  # 'UK' or 'USA'
        if not os.path.isdir(data_dir):
            print(f"Warning: Directory not found: {data_dir}")
            continue

        for gender in os.listdir(data_dir):
            gender_dir = os.path.join(data_dir, gender)
            if not os.path.isdir(gender_dir):
                continue

            for speaker in os.listdir(gender_dir):
                speaker_dir = os.path.join(gender_dir, speaker)
                if not os.path.isdir(speaker_dir):
                    continue

                speaker_id = f"{region}_{gender}_{speaker}"

                for filename in os.listdir(speaker_dir):
                    if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
                        continue

                    filepath = os.path.join(speaker_dir, filename)

                    # Label by filename convention
                    if filename.lower().startswith('original'):
                        label = 1  # real
                    elif filename.lower().startswith('synthetic'):
                        label = 0  # fake
                    else:
                        continue  # skip unrecognized files

                    filepaths.append(filepath)
                    labels.append(label)
                    speaker_ids.append(speaker_id)

    return filepaths, labels, speaker_ids


def train_val_split(filepaths, labels, speaker_ids, val_ratio=0.2, seed=42):
    """
    Split data by speaker to prevent data leakage.
    All files from a speaker go entirely into train OR val.
    """
    random.seed(seed)

    unique_speakers = list(set(speaker_ids))
    random.shuffle(unique_speakers)

    val_count = max(1, int(len(unique_speakers) * val_ratio))
    val_speakers = set(unique_speakers[:val_count])

    train_files, train_labels = [], []
    val_files, val_labels = [], []

    for fp, lbl, sid in zip(filepaths, labels, speaker_ids):
        if sid in val_speakers:
            val_files.append(fp)
            val_labels.append(lbl)
        else:
            train_files.append(fp)
            train_labels.append(lbl)

    return train_files, train_labels, val_files, val_labels


def get_dataloaders(data_dirs=None, batch_size=16, val_ratio=0.2):
    """
    Build train and validation DataLoaders from UK/ and USA/ directories.
    
    Args:
        data_dirs: list of paths to region directories (e.g. ['UK', 'USA'])
        batch_size: batch size for DataLoaders
        val_ratio: fraction of speakers to hold out for validation
    """
    if data_dirs is None:
        # Default: look for UK and USA in the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dirs = [
            os.path.join(script_dir, "UK"),
            os.path.join(script_dir, "USA"),
        ]

    filepaths, labels, speaker_ids = collect_audio_files(data_dirs)

    if len(filepaths) == 0:
        raise FileNotFoundError(f"No audio files found in: {data_dirs}")

    train_files, train_labels, val_files, val_labels = train_val_split(
        filepaths, labels, speaker_ids, val_ratio=val_ratio
    )

    train_dataset = AudioDataset(train_files, train_labels)
    val_dataset = AudioDataset(val_files, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dirs = [os.path.join(script_dir, "UK"), os.path.join(script_dir, "USA")]
    filepaths, labels, speaker_ids = collect_audio_files(dirs)

    print(f"Total files found: {len(filepaths)}")
    print(f"  Real (original): {labels.count(1)}")
    print(f"  Fake (synthetic): {labels.count(0)}")
    print(f"  Unique speakers: {len(set(speaker_ids))}")

    train_f, train_l, val_f, val_l = train_val_split(filepaths, labels, speaker_ids)
    print(f"\nTrain split: {len(train_f)} files ({train_l.count(1)} real, {train_l.count(0)} fake)")
    print(f"Val split:   {len(val_f)} files ({val_l.count(1)} real, {val_l.count(0)} fake)")

    print("\n--- Loading a batch ---")
    train_loader, val_loader = get_dataloaders()
    specs, lbls = next(iter(train_loader))
    print(f"Batch spectrogram shape: {specs.shape}")
    print(f"Batch labels: {lbls.tolist()}")
