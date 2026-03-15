import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
import os
from audio_model.model import get_model
from audio_model.dataset import load_audio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_audio(filepath, sample_rate=16000, fixed_length=3*16000):
    """
    Load and process an audio file into a log mel spectrogram tensor for inference.
    """
    waveform, sr = load_audio(filepath, sample_rate)
    
    # Resample if necessary
    if sr != sample_rate:
        resampler = T.Resample(sr, sample_rate)
        waveform = resampler(waveform)
        
    # Convert stereo to mono if necessary
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # Pad or truncate to fixed length
    if waveform.shape[1] > fixed_length:
        waveform = waveform[:, :fixed_length]
    elif waveform.shape[1] < fixed_length:
        pad_amount = fixed_length - waveform.shape[1]
        waveform = nn.functional.pad(waveform, (0, pad_amount))
        
    # Extract mel spectrogram
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    mel_spec = mel_spectrogram(waveform)
    
    # Convert to log scale (dB)
    log_mel_spec = T.AmplitudeToDB()(mel_spec)
    
    # Normalize
    log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-6)
    
    # Add batch dimension for CNN: [1, 1, 64, Time]
    log_mel_spec = log_mel_spec.unsqueeze(0)
    
    return log_mel_spec


def predict(audio_path, model_path="saved_models/best_model.pth"):
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return
        
    # Load model
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    
    # Process audio
    input_tensor = process_audio(audio_path).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
    # Print results
    classes = ['fake audio', 'real audio']
    print(f"--- Inference Results for: {os.path.basename(audio_path)} ---")
    for i, cls in enumerate(classes):
        print(f"'{cls}': {probabilities[0, i].item():.4f}")
        
    pred_class = classes[torch.argmax(probabilities, dim=1).item()]
    print(f"Prediction: {pred_class}")
    
    # Return the probability of it being "fake audio" (fraud)
    return probabilities[0, 0].item()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Voice Cloning Detection Inference")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file (wav, mp3)")
    parser.add_argument("--model", type=str, default="saved_models/best_model.pth", help="Path to trained model weights")
    
    args = parser.parse_args()
    
    if os.path.exists(args.audio):
        predict(args.audio, args.model)
    else:
        print(f"File not found: {args.audio}")
