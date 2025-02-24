# Aria API Documentation

## Core Components

### CreativeAgent

The main agent class that orchestrates all creative capabilities.

```python
from aria import CreativeAgent

agent = CreativeAgent(config=AgentConfig())
```

#### Configuration

```python
AgentConfig(
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    max_length: int = 1000,
    memory_size: int = 10000,
    personality: str = "creative",
    voice_enabled: bool = True,
    image_enabled: bool = True
)
```

#### Methods

##### process_input
```python
response = agent.process_input(
    text: Optional[str] = None,
    voice: Optional[bytes] = None,
    image: Optional[bytes] = None
) -> Dict
```

Process multimodal input and generate appropriate responses.

### StoryGenerator

Module for generating creative narratives and story elements.

```python
from aria.modules import StoryGenerator

generator = StoryGenerator()
```

#### Methods

##### generate
```python
story = generator.generate(
    theme: str,
    length: str = "medium",
    style: str = "default",
    context: Optional[List[str]] = None
) -> str
```

Generate a complete story based on given parameters.

##### generate_outline
```python
outline = generator.generate_outline(
    theme: str,
    num_chapters: int = 3
) -> List[str]
```

Generate a story outline with chapter descriptions.

##### generate_character
```python
character = generator.generate_character(
    story_theme: str
) -> Dict[str, str]
```

Generate a detailed character profile.

### MusicGenerator

Module for creating musical compositions.

```python
from aria.modules import MusicGenerator

composer = MusicGenerator()
```

#### Methods

##### compose
```python
midi_data = composer.compose(
    genre: str = "electronic",
    mood: str = "uplifting",
    duration: int = 60,
    key: str = "C",
    scale: str = "major"
) -> bytes
```

Compose a complete musical piece.

##### export_to_file
```python
composer.export_to_file(
    midi_data: bytes,
    filename: str
)
```

Export MIDI data to a file.

### ArtDirector

Module for generating art prompts and style guides.

```python
from aria.modules import ArtDirector

director = ArtDirector()
```

#### Methods

##### generate_prompt
```python
prompt = director.generate_prompt(
    style: str,
    subject: str,
    medium: str = "digital",
    mood: Optional[str] = None,
    additional_details: Optional[Dict] = None
) -> str
```

Generate a detailed art prompt.

##### generate_style_guide
```python
guide = director.generate_style_guide(
    style: str
) -> Dict[str, str]
```

Generate a comprehensive style guide.

### EmotionRecognizer

Module for analyzing emotional content from different inputs.

```python
from aria.modules import EmotionRecognizer

recognizer = EmotionRecognizer()
```

#### Methods

##### from_text
```python
emotions = recognizer.from_text(
    text: str
) -> Dict[str, float]
```

Recognize emotions from text input.

##### from_voice
```python
emotions = recognizer.from_voice(
    audio_data: bytes
) -> Dict[str, float]
```

Recognize emotions from voice input.

##### from_multimodal
```python
emotions = recognizer.from_multimodal(
    text: Optional[str] = None,
    voice: Optional[bytes] = None
) -> Dict[str, float]
```

Combine text and voice emotion recognition.

### Memory

Advanced memory system for contextual information.

```python
from aria import Memory

memory = Memory(capacity=10000)
```

#### Methods

##### store
```python
memory.store(
    input_text: str,
    response: str,
    emotion: Optional[str] = None,
    context: Optional[Dict] = None
)
```

Store a new memory entry.

##### get_relevant_memories
```python
memories = memory.get_relevant_memories(
    query: str,
    k: int = 5
) -> List[Dict]
```

Retrieve relevant memories based on semantic similarity.

## Usage Examples

### Creative Story Generation
```python
from aria import CreativeAgent

agent = CreativeAgent()

# Generate a cyberpunk story
story = agent.generate_story(
    theme="cyberpunk rebellion",
    length="medium",
    style="noir"
)
print(story)
```

### Musical Composition
```python
from aria.modules import MusicGenerator

composer = MusicGenerator()

# Create an uplifting electronic track
melody = composer.compose(
    genre="electronic",
    mood="uplifting",
    duration=120  # 2 minutes
)
composer.export_to_file(melody, "composition.midi")
```

### Art Direction
```python
from aria.modules import ArtDirector

director = ArtDirector()

# Generate a surrealist art prompt
prompt = director.generate_prompt(
    style="surrealism",
    subject="dreams and consciousness",
    medium="digital",
    mood="mysterious"
)
print(prompt)
```

### Emotion Analysis
```python
from aria.modules import EmotionRecognizer

recognizer = EmotionRecognizer()

# Analyze text emotion
text = "I'm so excited about this new project!"
emotions = recognizer.from_text(text)
print(emotions)
```

## Error Handling

All modules include comprehensive error handling. Here's an example:

```python
try:
    story = agent.generate_story(theme="adventure")
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"Generation error: {e}")
```

## Best Practices

1. **Memory Management**
   - Regularly clear unused memories
   - Monitor memory usage statistics

2. **Performance Optimization**
   - Use batch processing when possible
   - Enable GPU acceleration when available

3. **Error Handling**
   - Always wrap API calls in try-except blocks
   - Implement proper fallback mechanisms

4. **Resource Management**
   - Close file handles after use
   - Release GPU memory when done

## Advanced Configuration

### Environment Variables
```bash
ARIA_MODEL_PATH=/path/to/models
ARIA_API_KEY=your_api_key
ARIA_MAX_MEMORY=10000
```

### Configuration File
```yaml
# config.yaml
model:
  name: gpt-3.5-turbo
  temperature: 0.7
memory:
  capacity: 10000
  importance_threshold: 0.5
```
