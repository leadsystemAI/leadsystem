from typing import Dict, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..utils.templates import STORY_TEMPLATES

class StoryGenerator:
    """
    Story generation module that creates engaging narratives with rich character
    development and compelling plot structures.
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
        # Story generation parameters
        self.length_tokens = {
            "short": 500,
            "medium": 1000,
            "long": 2000
        }
        
        # Load story templates
        self.templates = STORY_TEMPLATES

    def generate(self, 
                theme: str, 
                length: str = "medium", 
                style: str = "default",
                context: Optional[List[str]] = None) -> str:
        """
        Generate a story based on given parameters.
        
        Args:
            theme (str): Main theme or topic of the story
            length (str): Desired length ('short', 'medium', 'long')
            style (str): Writing style ('default', 'noir', 'fantasy', etc.)
            context (List[str], optional): Previous context or background information
            
        Returns:
            str: Generated story
        """
        # Build prompt
        prompt = self._build_prompt(theme, style, context)
        
        # Generate story
        max_length = self.length_tokens.get(length, 1000)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            temperature=0.8,
            top_p=0.9,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        story = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._post_process(story)

    def _build_prompt(self, theme: str, style: str, context: Optional[List[str]]) -> str:
        """
        Build a prompt for story generation.
        """
        template = self.templates.get(style, self.templates["default"])
        
        prompt_parts = [
            template["prefix"],
            f"Theme: {theme}",
            f"Style: {style}",
        ]
        
        if context:
            prompt_parts.append("Background:")
            prompt_parts.extend(context)
            
        prompt_parts.append(template["suffix"])
        
        return "\n".join(prompt_parts)

    def _post_process(self, story: str) -> str:
        """
        Clean and format the generated story.
        """
        # Remove any remaining prompt text
        story = story.split("Story:")[-1].strip()
        
        # Format paragraphs
        paragraphs = story.split("\n\n")
        formatted_paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return "\n\n".join(formatted_paragraphs)

    def generate_outline(self, theme: str, num_chapters: int = 3) -> List[str]:
        """
        Generate a story outline with chapter descriptions.
        """
        prompt = f"Create a {num_chapters}-chapter outline for a story about {theme}:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=500,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        outline = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        chapters = outline.split("Chapter")[1:]  # Remove prompt
        return [chapter.strip() for chapter in chapters]

    def generate_character(self, story_theme: str) -> Dict[str, str]:
        """
        Generate a detailed character profile.
        """
        prompt = f"Create a character profile for a story about {story_theme}:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=300,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        profile_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse profile into structured data
        lines = profile_text.split("\n")
        profile = {}
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                profile[key.strip()] = value.strip()
                
        return profile
