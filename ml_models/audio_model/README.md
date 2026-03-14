# 🔊 Audio Fraud Detection Model — Voice Cloning Detector

## 📌 What This Project Does

This model answers one question: **"Is this audio clip a real human voice, or was it generated/cloned by AI?"**

Voice cloning tools (like ElevenLabs, Bark, Tortoise-TTS) can now create hyper-realistic fake voices. This model detects those fakes by analyzing the **hidden patterns in the audio signal** that humans can't hear but AI leaves behind.

---

## 🧠 Core Concept: Why Spectrograms?

### The Problem With Raw Audio
A raw audio file is just a list of numbers — amplitude values sampled thousands of times per second. For a 3-second clip at 16kHz, that's **48,000 numbers in a single row**. Feeding this directly into a neural network would be:
- Extremely slow (too many input features)
- Ineffective (the network can't easily spot patterns in a 1D stream)

### The Solution: Convert Sound → Image
Instead of processing raw audio, we convert it into a **Mel Spectrogram** — a 2D image-like representation of sound.

```
Raw Audio Waveform          →    Mel Spectrogram (2D image)
[0.1, -0.3, 0.5, ...]           ┌──────────────────┐
(48,000 numbers in a row)        │ ░░▓▓██░░▓▓██░░▓▓ │  ← high frequency
                                 │ ▓▓██░░▓▓██░░▓▓██ │
                                 │ ██░░▓▓██░░▓▓████ │  ← low frequency
                                 └──────────────────┘
                                   time →
```

**What a Mel Spectrogram shows:**
- **X-axis** = Time (left to right, the audio progresses)
- **Y-axis** = Frequency (bottom = low/bass, top = high/treble)
- **Brightness** = Energy (how loud that frequency is at that moment)

**Why "Mel"?** The Mel scale mimics how humans hear pitch. We're more sensitive to differences at lower frequencies (e.g., 200Hz vs 300Hz sounds very different) than higher ones (e.g., 8000Hz vs 8100Hz barely differs). The Mel scale compresses high frequencies to match this perception.

### Why This Works for Fake Detection
AI-generated voices leave subtle fingerprints in their spectrograms:
- **Unnaturally smooth frequency transitions** (real voices have micro-variations)
- **Missing or artificial harmonics** (human vocal cords create complex overtone patterns)
- **Vocoder artifacts** (the neural network that generates the final audio waveform leaves repeating patterns)

These patterns are invisible when listening but visible in spectrograms — which is exactly what our CNN learns to detect.

---

## 📊 Our Dataset

### Source
We use voice samples from **UK** and **USA** speakers, organized by region and gender:

```text
audio_model/
├── UK/
│   ├── female/
│   │   ├── 1/                        ← Speaker 1
│   │   │   ├── original.m4a          ← Real human voice recording
│   │   │   ├── synthetic_1.mp3       ← AI clone attempt #1
│   │   │   ├── synthetic_2.mp3       ← AI clone attempt #2
│   │   │   └── synthetic_3.mp3       ← AI clone attempt #3
│   │   ├── 2/ ...                    ← Speaker 2
│   │   └── 5/ ...                    ← Speaker 5
│   └── male/
│       ├── 1/ ... → 5/ ...
└── USA/
    ├── female/
    │   ├── 1/ ... → 5/ ...
    └── male/
        ├── 1/ ... → 5/ ...
```

### Numbers
| Metric | Count |
|--------|-------|
| Total audio files | **80** |
| Real (original) files | 20 (one per speaker) |
| Fake (synthetic) files | 60 (three per speaker) |
| Unique speakers | 20 (5M + 5F per region × 2 regions) |
| Audio formats | `.m4a`, `.wav`, `.mp3` |

### How Labels Are Assigned
Instead of organizing files into `fake/` and `real/` folders (like the vision model does), our data labels are determined by **filename**:
- `original.*` → **Real** (label `1`)
- `synthetic_*.*` → **Fake** (label `0`)

This is handled automatically by `collect_audio_files()` in `dataset.py`.

---

## 🎧 The Feature Extraction Pipeline (dataset.py)

Each audio file goes through **6 processing steps** before it enters the model. Here's what each step does and *why*:

### Step 1: Load Audio (via ffmpeg)
```python
waveform, sr = load_audio(filepath, self.sample_rate)
```
**What:** Reads the audio file and converts it to a numerical array.

**Technical Detail:** We use `ffmpeg` (bundled via `imageio-ffmpeg`) instead of `torchaudio.load` because Python 3.13 removed the `audioop` module, breaking most audio library backends. Our `load_audio()` function pipes the audio through ffmpeg → converts to WAV → reads with Python's built-in `wave` module.

### Step 2: Resample to 16kHz
```python
if sr != self.sample_rate:
    resampler = T.Resample(sr, self.sample_rate)
    waveform = resampler(waveform)
```
**What:** Converts all audio to 16,000 samples per second.

**Why:** Different recordings may be at different sample rates (44.1kHz for music, 48kHz for video, etc.). We standardize to **16kHz** because:
- It's the speech processing standard (captures all human voice frequencies up to 8kHz by Nyquist theorem)
- It reduces data size without losing relevant speech information
- The model expects a consistent input format

### Step 3: Convert to Mono
```python
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)
```
**What:** If the audio has 2 channels (stereo), average them into 1 (mono).

**Why:** We're detecting voice cloning artifacts, not spatial audio. Stereo adds complexity without useful information for this task.

### Step 4: Pad or Truncate to Fixed Length (3 seconds)
```python
if waveform.shape[1] > self.fixed_length:       # Too long → cut
    waveform = waveform[:, :self.fixed_length]
elif waveform.shape[1] < self.fixed_length:      # Too short → pad with zeros
    waveform = nn.functional.pad(waveform, (0, pad_amount))
```
**What:** Force all clips to exactly **48,000 samples** (3 seconds × 16,000 samples/sec).

**Why:** Neural networks need fixed-size inputs. We chose 3 seconds because:
- It's enough to capture voice characteristics and cloning artifacts
- It keeps computation manageable
- Most cloning artifacts appear within the first few seconds

### Step 5: Compute Mel Spectrogram
```python
self.mel_spectrogram = T.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,       # Window size for FFT
    hop_length=512,   # Step size between windows
    n_mels=64         # Number of Mel frequency bins
)
mel_spec = self.mel_spectrogram(waveform)
```
**What:** Converts the 1D waveform into a 2D time-frequency image.

**How it works (simplified):**
1. Slide a window of 1024 samples across the audio, moving 512 samples each step
2. For each window position, compute the **FFT** (Fast Fourier Transform) to find which frequencies are present
3. Group these frequencies into 64 "Mel bins" (perceptually spaced)
4. The result is a 2D matrix: **64 frequency bins × 94 time steps**

**Key parameters explained:**
- `n_fft=1024`: Larger windows = better frequency resolution but worse time resolution (tradeoff)
- `hop_length=512`: How far the window slides each step. 512 = 50% overlap with the 1024 window, giving smooth transitions
- `n_mels=64`: 64 frequency bands. More bands = finer detail but larger input. 64 is a common sweet spot for speech

**Why 94 time steps?** With 48,000 samples, hop_length=512: `48000 / 512 ≈ 94` windows fit across the audio.

**Output shape:** `[1, 64, 94]` → 1 channel × 64 frequency bins × 94 time frames

### Step 6: Log Scale + Normalize
```python
log_mel_spec = T.AmplitudeToDB()(mel_spec)
log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-6)
```
**What:**
1. Convert power values to **decibels (dB)** — logarithmic scale
2. **Standardize** to zero mean and unit variance

**Why log scale?** Human hearing is logarithmic. The difference between 10W and 20W of sound power is perceptually the same as 100W vs 200W. dB captures this. Without it, quiet sounds would be invisible to the model and loud sounds would dominate.

**Why normalize?** Neural networks train faster and more stably when inputs are centered around 0 with similar scales. The `+1e-6` prevents division by zero for silent clips.

---

## 🏗 The Model Architecture (model.py)

We use a **custom CNN (Convolutional Neural Network)** called `AudioCNN`. Here's a layer-by-layer breakdown:

### Why a CNN for Audio?

Since we converted audio into a 2D spectrogram (which looks like an image), we can use the same CNN techniques that work for image classification. The CNN learns to detect **local patterns** in the spectrogram — exactly the kind of artifacts that AI voice cloning produces.

### Architecture Diagram

```
Input: [batch, 1, 64, 94]    (spectrogram "image")
         │
    ┌────▼────┐
    │ Conv2d  │  1→16 filters, 3×3 kernel
    │BatchNorm│
    │  ReLU   │
    │MaxPool2d│  2×2 → halves dimensions
    └────┬────┘  Output: [batch, 16, 32, 47]
         │
    ┌────▼────┐
    │ Conv2d  │  16→32 filters
    │BatchNorm│
    │  ReLU   │
    │MaxPool2d│
    └────┬────┘  Output: [batch, 32, 16, 23]
         │
    ┌────▼────┐
    │ Conv2d  │  32→64 filters
    │BatchNorm│
    │  ReLU   │
    │MaxPool2d│
    └────┬────┘  Output: [batch, 64, 8, 11]
         │
    ┌────▼────┐
    │ Conv2d  │  64→128 filters
    │BatchNorm│
    │  ReLU   │
    │MaxPool2d│
    └────┬────┘  Output: [batch, 128, 4, 5]
         │
    ┌────▼────┐
    │Adaptive │
    │AvgPool  │  → [batch, 128, 1, 1]
    └────┬────┘
         │
    ┌────▼────┐
    │Flatten  │  → [batch, 128]
    │Linear   │  128 → 64
    │  ReLU   │
    │Dropout  │  50% (prevents overfitting)
    │Linear   │  64 → 2 (fake, real)
    └────┬────┘
         │
    Output: [batch, 2]    (logits for fake vs real)
```

### Key Components Explained

**Conv2d (Convolutional Layer):**
Slides small filters (3×3 pixels) across the spectrogram to detect patterns. Early layers detect simple edges and textures; deeper layers detect complex patterns like vocoder artifacts or unnatural frequency transitions.

**BatchNorm (Batch Normalization):**
Normalizes activations within each mini-batch. This stabilizes and speeds up training by preventing internal values from drifting too high or low.

**ReLU (Rectified Linear Unit):**
Activation function: `output = max(0, input)`. Introduces non-linearity — without it, stacking layers wouldn't help because linear operations compose into one linear operation.

**MaxPool2d (2×2):**
Takes the maximum value in each 2×2 block, halving the spatial dimensions. This makes the model invariant to small shifts in the spectrogram and reduces computation.

**AdaptiveAvgPool2d(1, 1):**
Averages all remaining spatial dimensions into a single value per channel. This is critical because it means the model can handle **any input size** — if you change the audio length, the spectrogram width changes, but this layer adapts.

**Dropout(0.5):**
During training, randomly zeroes out 50% of neurons. This forces the network to not rely on any single feature, dramatically reducing overfitting (especially important with only 80 samples).

---

## 🏋️ The Training Loop (train.py)

### How Training Works

```
For each epoch (full pass through data):
    1. TRAINING PHASE:
       - Feed batches of spectrograms through the model
       - Compare predictions to true labels (CrossEntropyLoss)
       - Backpropagate the error (compute gradients)
       - Update weights (Adam optimizer)
       - Track accuracy and loss

    2. VALIDATION PHASE:
       - Feed validation data through the model (no gradient computation)
       - Measure accuracy on data the model hasn't trained on
       - Save model if this is the best validation accuracy so far
```

### Key Training Concepts

**CrossEntropyLoss:**
The standard loss function for classification. It measures how far the model's predicted probabilities are from the true labels. Lower = better. It penalizes confident wrong predictions heavily.

**Adam Optimizer (lr=1e-4):**
Adam adapts the learning rate for each parameter individually. `lr=1e-4` (0.0001) is a conservative learning rate — the model takes small, careful steps to avoid overshooting good solutions.

**Why we use `model.train()` vs `model.eval()`:**
- `model.train()`: BatchNorm uses batch statistics; Dropout is active
- `model.eval()`: BatchNorm uses running averages; Dropout is disabled
- Forgetting to switch modes is a common bug that causes inconsistent results

**Why `torch.no_grad()` during validation:**
Disables gradient computation, saving memory and computation. We don't need gradients because we're not updating weights during validation — just measuring performance.

### Train/Validation Split Strategy

```python
# ❌ WRONG: Random file split
# If Speaker 1's original.m4a is in TRAIN and synthetic_1.mp3 is in VAL,
# the model might learn Speaker 1's voice characteristics rather than
# "real vs fake" differences. This is called DATA LEAKAGE.

# ✅ CORRECT: Speaker-level split (what we do)
# All 4 files from Speaker 1 go into TRAIN or VAL together.
# The model must generalize to unseen speakers.
```

This is handled by `train_val_split()` in `dataset.py`, which groups files by speaker ID (e.g., `UK_female_1`) and assigns entire speakers to either train or validation.

---

## 🔍 Inference (inference.py)

The inference script lets you classify any new audio file:

```bash
python inference.py --audio path/to/voice.mp3
```

**What happens internally:**
1. The audio file goes through the same 6-step pipeline as training data
2. The processed spectrogram is fed through the trained model
3. `softmax` converts raw logits to probabilities (summing to 1.0)
4. The class with the highest probability is the prediction

**Example output:**
```
--- Inference Results for: suspicious_call.mp3 ---
'fake audio': 0.8723
'real audio': 0.1277
Prediction: fake audio
```

---

## 🗂 Complete Project Structure

```text
audio_model/
│
├── UK/                              # UK voice samples
│   ├── female/1-5/                  #   5 female speakers
│   └── male/1-5/                    #   5 male speakers
│
├── USA/                             # USA voice samples
│   ├── female/1-5/                  #   5 female speakers
│   └── male/1-5/                    #   5 male speakers
│
├── saved_models/                    # Created after training
│   └── best_model.pth              #   Best model weights
│
├── dataset.py                       # Data loading + feature extraction
│   ├── load_audio()                 #   ffmpeg-based universal audio loader
│   ├── AudioDataset                 #   PyTorch Dataset (spectrogram pipeline)
│   ├── collect_audio_files()        #   Walks UK/USA dirs, assigns labels
│   ├── train_val_split()            #   Speaker-level train/val split
│   └── get_dataloaders()            #   Returns ready-to-use DataLoaders
│
├── model.py                         # Neural network definition
│   └── AudioCNN                     #   4-block CNN + GAP + FC classifier
│
├── train.py                         # Training script
│   └── Training loop                #   10 epochs, saves best model
│
├── inference.py                     # Prediction script
│   ├── process_audio()              #   Single-file spectrogram extraction
│   └── predict()                    #   Load model → classify → print results
│
├── dataset_voice_changing_metainfo.xlsx  # Metadata spreadsheet
└── README.md                        # This file
```

---

## 📦 Dependencies

| Package | Why We Need It |
|---------|----------------|
| `torch` | Core deep learning framework — tensors, neural networks, training |
| `torchaudio` | Audio-specific transforms (MelSpectrogram, AmplitudeToDB, Resample) |
| `imageio-ffmpeg` | Bundles an ffmpeg binary to decode .m4a/.mp3 without system installs |
| `tqdm` | Progress bars during training (visual feedback for long loops) |
| `numpy` | Converting raw audio bytes to numerical arrays |

```bash
pip install torch torchaudio imageio-ffmpeg tqdm numpy
```

---

## 🚀 Quick Start

```bash
# 1. Verify the dataset loads correctly
python dataset.py

# 2. Train the model
python train.py

# 3. Run inference on a new audio file
python inference.py --audio path/to/audio.mp3
```

---

## 🔬 Future Improvements

| Improvement | Why It Helps |
|-------------|--------------|
| **Data Augmentation** (time stretch, pitch shift, noise) | More training variety from limited samples |
| **Transfer Learning** (Wav2Vec 2.0, HuBERT) | Pre-trained audio models already understand speech patterns |
| **Class Weighting** | Our dataset is 3:1 imbalanced (60 fake vs 20 real) |
| **ROC-AUC Metrics** | Better evaluation than accuracy for imbalanced datasets |
| **Cross-Validation** | More reliable accuracy estimates with small datasets |
| **Larger Dataset** (ASVspoof, WaveFake) | Better generalization to unseen cloning methods |

---

## 🎯 Current Status

✔ Full data pipeline (UK + USA, 80 audio files)  
✔ Speaker-level train/val split (prevents data leakage)  
✔ Universal audio format support (.wav, .mp3, .m4a, .flac)  
✔ GPU-accelerated training  
✔ CLI inference tool  
✔ Production-ready baseline model  
