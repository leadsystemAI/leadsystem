from typing import Dict, List, Optional, Union
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from dataclasses import dataclass
from .modules.story import StoryGenerator
from .modules.music import MusicGenerator
from .modules.art import ArtDirector
from .modules.emotion import EmotionRecognizer
from .memory import Memory
 
@dataclass
class AgentConfig:
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_length: int = 1000
    memory_size: int = 10000
    personality: str = "creative"
    voice_enabled: bool = True
    image_enabled: bool = True

class CreativeAgent:
    """
    Aria's core agent class that combines multiple creative capabilities
    into a cohesive AI assistant.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.story_generator = StoryGenerator()
        self.music_generator = MusicGenerator()
        self.art_director = ArtDirector()
        self.emotion_recognizer = EmotionRecognizer()
        self.memory = Memory(capacity=self.config.memory_size)
        
        # Initialize language model
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name).to(self.device)

    def generate_story(self, theme: str, length: str = "medium", style: str = "default") -> str:
        """
        Generate a creative story based on given parameters.
        """
        context = self.memory.get_relevant_memories(theme)
        return self.story_generator.generate(
            theme=theme,
            length=length,
            style=style,
            context=context
        )

    def compose_music(self, genre: str, mood: str, duration: int = 60) -> bytes:
        """
        Compose a musical piece based on given parameters.
        """
        return self.music_generator.compose(
            genre=genre,
            mood=mood,
            duration=duration
        )

    def generate_art_prompt(self, style: str, subject: str, medium: str = "digital") -> str:
        """
        Generate a detailed art prompt for image creation.
        """
        return self.art_director.generate_prompt(
            style=style,
            subject=subject,
            medium=medium
        )

    def process_input(self, 
                     text: Optional[str] = None, 
                     voice: Optional[bytes] = None,
                     image: Optional[bytes] = None) -> Dict:
        """
        Process multimodal input and generate appropriate response.
        """
        response = {}
        emotion = None

        if voice and self.config.voice_enabled:
            text_from_voice = self._process_voice(voice)
            emotion = self.emotion_recognizer.from_voice(voice)
            response["voice_text"] = text_from_voice

        if image and self.config.image_enabled:
            image_description = self._process_image(image)
            response["image_description"] = image_description

        if text:
            if emotion is None:
                emotion = self.emotion_recognizer.from_text(text)
            
            response["reply"] = self._generate_response(
                text=text,
                emotion=emotion,
                context=response
            )

        return response

    def _process_voice(self, voice_data: bytes) -> str:
        """
        Convert voice input to text and analyze emotional content.
        """
        # Voice processing implementation
        pass

    def _process_image(self, image_data: bytes) -> str:
        """
        Analyze image content and generate description.
        """
        # Image processing implementation
        pass

    def _generate_response(self, text: str, emotion: Optional[str], context: Dict) -> str:
        """ 
        Generate contextually and emotionally appropriate response.
        """
        prompt = self._build_prompt(text, emotion, context)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=self.config.max_length,
            temperature=self.config.temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Store interaction in memory
        self.memory.store(
            input_text=text,
            response=response,
            emotion=emotion,
            context=context
        )
        
        return response

    def _build_prompt(self, text: str, emotion: Optional[str], context: Dict) -> str:
        """
        Build a prompt that includes context and emotional awareness.
        """
        relevant_memories = self.memory.get_relevant_memories(text)
        
        prompt_parts = [
            f"Previous context: {relevant_memories}",
            f"Current emotion: {emotion}" if emotion else "",
            f"User input: {text}",
            "Assistant: "
        ]
        
        return "\n".join(filter(None, prompt_parts))

    def save_state(self, path: str):
        """
        Save agent's state and memory.
        """
        torch.save({
            'config': self.config,
            'memory': self.memory.state_dict(),
            'model_state': self.model.state_dict()
        }, path)

    def load_state(self, path: str):
        """
        Load agent's state and memory.
        """
        checkpoint = torch.load(path)
        self.config = checkpoint['config']
        self.memory.load_state_dict(checkpoint['memory'])
        self.model.load_state_dict(checkpoint['model_state'])
