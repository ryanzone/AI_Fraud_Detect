import sys
import os

# Add parent directory to sys.path to allow importing from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import specific prediction functions (they should handle lazy loading or be initialized on import)
try:
    from text_model.predict import predict_text
except ImportError as e:
    print(f"Warning: Could not import text_model: {e}")
    predict_text = None

try:
    from vision_model.inference import predict_image
except ImportError as e:
    print(f"Warning: Could not import vision_model: {e}")
    predict_image = None

try:
    from audio_model.inference import predict as predict_audio
except ImportError as e:
    print(f"Warning: Could not import audio_model: {e}")
    predict_audio = None


class FusionEngine:
    def __init__(self, weights=None):
        """
        weights: dict of modality weights, e.g. {'text': 0.4, 'vision': 0.3, 'audio': 0.3}
        """
        if weights is None:
            self.weights = {'text': 0.334, 'vision': 0.333, 'audio': 0.333}
        else:
            self.weights = weights

    def predict(self, text=None, image_path=None, audio_path=None):
        """
        Runs inference on provided modalities and returns a fused fraud probability.
        """
        results = {}
        used_weights = {}

        if text is not None and predict_text is not None:
            text_prob = predict_text(text)
            results['text_fraud_prob'] = text_prob
            used_weights['text'] = self.weights['text']

        if image_path is not None and os.path.exists(image_path) and predict_image is not None:
            vision_prob = predict_image(image_path)
            results['vision_fraud_prob'] = vision_prob
            used_weights['vision'] = self.weights['vision']

        if audio_path is not None and os.path.exists(audio_path) and predict_audio is not None:
            # Note: the audio model predict function might need the model path argument if default isn't enough
            # By default it uses saved_models/best_model.pth relative to its execution, we might need a fixed path
            audio_script_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'audio_model')
            default_audio_model = os.path.join(audio_script_dir, "saved_models", "best_model.pth")
            
            audio_prob = predict_audio(audio_path, model_path=default_audio_model)
            if audio_prob is not None:
                results['audio_fraud_prob'] = audio_prob
                used_weights['audio'] = self.weights['audio']

        # Calculate final fusion probability
        if not results:
            return {"error": "No valid modalities provided or models failed to load.", "fusion_fraud_prob": 0.0}

        # Normalize weights for the provided modalities
        total_weight = sum(used_weights.values())
        if total_weight == 0:
            return {"error": "Total weight is 0.", "fusion_fraud_prob": 0.0}

        fusion_prob = 0.0
        for modality, prob in results.items():
            modality_name = modality.split('_')[0]
            weight = used_weights[modality_name] / total_weight
            fusion_prob += prob * weight

        results['fusion_fraud_prob'] = fusion_prob
        
        # Add risk level
        if fusion_prob > 0.7:
            results['risk_level'] = "HIGH - Likely Fraudulent"
        elif fusion_prob > 0.4:
            results['risk_level'] = "MEDIUM - Suspicious"
        else:
            results['risk_level'] = "LOW - Likely Legitimate"

        return results
