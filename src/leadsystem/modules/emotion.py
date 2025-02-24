from typing import Dict, Optional
import torch
import numpy as np
import librosa
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ..utils.emotion_mapping import EMOTION_CATEGORIES

class EmotionRecognizer:
    """
    Emotion recognition module that analyzes emotional content from
    text, voice, and combined inputs.
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Text emotion model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(EMOTION_CATEGORIES)
        ).to(self.device)
        
        # Audio parameters
        self.sample_rate = 16000
        self.emotion_categories = EMOTION_CATEGORIES

    def from_text(self, text: str) -> Dict[str, float]:
        """
        Recognize emotions from text input.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, float]: Emotion probabilities
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        emotions = {
            emotion: prob.item()
            for emotion, prob in zip(self.emotion_categories, probs[0])
        }
        
        return emotions

    def from_voice(self, audio_data: bytes) -> Dict[str, float]:
        """
        Recognize emotions from voice input.
        
        Args:
            audio_data (bytes): Raw audio data
            
        Returns:
            Dict[str, float]: Emotion probabilities
        """
        # Convert audio data to numpy array
        audio = self._bytes_to_audio(audio_data)
        
        # Extract acoustic features
        features = self._extract_acoustic_features(audio)
        
        # Predict emotions using acoustic features
        emotions = self._predict_emotions_from_features(features)
        
        return emotions

    def from_multimodal(self, 
                       text: Optional[str] = None,
                       voice: Optional[bytes] = None) -> Dict[str, float]:
        """
        Combine text and voice emotion recognition.
        
        Args:
            text (str, optional): Input text
            voice (bytes, optional): Raw audio data
            
        Returns:
            Dict[str, float]: Combined emotion probabilities
        """
        emotions = {emotion: 0.0 for emotion in self.emotion_categories}
        
        if text:
            text_emotions = self.from_text(text)
            for emotion, prob in text_emotions.items():
                emotions[emotion] += prob * 0.6  # Text weight: 60%
                
        if voice:
            voice_emotions = self.from_voice(voice)
            for emotion, prob in voice_emotions.items():
                emotions[emotion] += prob * 0.4  # Voice weight: 40%
                
        # Normalize probabilities
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v/total for k, v in emotions.items()}
            
        return emotions

    def _bytes_to_audio(self, audio_data: bytes) -> np.ndarray:
        """
        Convert audio bytes to numpy array.
        """
        import io
        import soundfile as sf
        
        audio_io = io.BytesIO(audio_data)
        audio, _ = sf.read(audio_io)
        
        # Resample if necessary
        if _ != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=_, target_sr=self.sample_rate)
            
        return audio

    def _extract_acoustic_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract acoustic features from audio signal.
        """
        # Extract various acoustic features
        features = []
        
        # Mel-frequency cepstral coefficients (MFCCs)
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        features.append(np.mean(mfccs, axis=1))
        
        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        features.append(np.mean(centroid))
        
        # Root Mean Square Energy
        rms = librosa.feature.rms(y=audio)
        features.append(np.mean(rms))
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        features.append(np.mean(zcr))
        
        return np.concatenate(features)

    def _predict_emotions_from_features(self, features: np.ndarray) -> Dict[str, float]:
        """
        Predict emotions from acoustic features.
        """
        # This is a simplified version. In practice, you would use a trained
        # model for audio emotion recognition.
        
        # Normalize features
        features = (features - np.mean(features)) / np.std(features)
        
        # Generate pseudo-probabilities based on feature patterns
        probs = np.abs(np.sin(features))  # Simplified example
        probs = probs / np.sum(probs)
        
        emotions = {
            emotion: prob
            for emotion, prob in zip(self.emotion_categories, probs)
        }
        
        return emotions

    def get_dominant_emotion(self, emotions: Dict[str, float]) -> str:
        """
        Get the dominant emotion from probability distribution.
        """
        return max(emotions.items(), key=lambda x: x[1])[0]

    def get_emotion_intensity(self, emotions: Dict[str, float]) -> float:
        """
        Calculate the intensity of the emotional response.
        """
        # Use the probability of the dominant emotion as intensity
        return max(emotions.values())
