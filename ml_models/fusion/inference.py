import argparse
import sys
import os
import json

# Add parent directory to sys.path to allow importing from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion.fusion_engine import FusionEngine

def main():
    parser = argparse.ArgumentParser(description="Multi-Modal Fraud Detection Fusion Inference")
    parser.add_argument("--text", type=str, help="Text to analyze")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--audio", type=str, help="Path to audio file")
    
    args = parser.parse_args()
    
    if not any([args.text, args.image, args.audio]):
        print("Please provide at least one modality (--text, --image, or --audio)")
        sys.exit(1)
        
    print("Loading models...")
    engine = FusionEngine()
    
    print("Running inference...")
    results = engine.predict(text=args.text, image_path=args.image, audio_path=args.audio)
    
    print("\n--- Fusion Model Results ---")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
