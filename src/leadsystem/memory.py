from typing import Dict, List, Optional, Any
import torch
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
 
@dataclass
class MemoryEntry:
    """
    Structure for storing individual memory entries.
    """
    timestamp: datetime
    input_text: str
    response: str
    emotion: Optional[str]
    context: Dict[str, Any]
    embedding: torch.Tensor
    importance: float
    access_count: int

class Memory:
    """
    Advanced memory system for storing and retrieving contextual information
    with importance-based management and semantic search.
    """
    
    def __init__(self, 
                 capacity: int = 10000,
                 model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.capacity = capacity
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize transformer model for embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Memory storage
        self.memories: List[MemoryEntry] = []
        
        # Memory management parameters
        self.importance_threshold = 0.5
        self.time_decay_factor = 0.1
        self.access_boost = 0.1

    def store(self,
              input_text: str,
              response: str,
              emotion: Optional[str] = None,
              context: Optional[Dict] = None) -> None:
        """
        Store a new memory entry.
        """
        # Generate embedding
        embedding = self._generate_embedding(input_text)
        
        # Calculate initial importance
        importance = self._calculate_importance(input_text, emotion)
        
        # Create memory entry
        entry = MemoryEntry(
            timestamp=datetime.now(),
            input_text=input_text,
            response=response,
            emotion=emotion,
            context=context or {},
            embedding=embedding,
            importance=importance,
            access_count=0
        )
        
        # Add to memory
        self.memories.append(entry)
        
        # Manage memory capacity
        if len(self.memories) > self.capacity:
            self._cleanup_memories()

    def get_relevant_memories(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve relevant memories based on semantic similarity.
        """
        if not self.memories:
            return []
            
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        
        # Calculate similarities
        similarities = []
        for memory in self.memories:
            similarity = torch.cosine_similarity(
                query_embedding,
                memory.embedding,
                dim=0
            )
            similarities.append(similarity.item())
            
        # Get top k memories
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        relevant_memories = []
        for idx in top_indices:
            memory = self.memories[idx]
            
            # Update access count and importance
            memory.access_count += 1
            memory.importance += self.access_boost
            
            relevant_memories.append({
                'input': memory.input_text,
                'response': memory.response,
                'emotion': memory.emotion,
                'context': memory.context,
                'similarity': similarities[idx]
            })
            
        return relevant_memories

    def _generate_embedding(self, text: str) -> torch.Tensor:
        """
        Generate embedding for text using transformer model.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)
            
        return embedding.squeeze()

    def _calculate_importance(self, text: str, emotion: Optional[str]) -> float:
        """
        Calculate importance score for a memory entry.
        """
        importance = 0.5  # Base importance
        
        # Length-based importance
        words = len(text.split())
        importance += min(words / 100, 0.2)  # Up to 0.2 for length
        
        # Emotion-based importance
        if emotion:
            importance += 0.2  # Emotional content is important
            
        return min(importance, 1.0)

    def _cleanup_memories(self) -> None:
        """
        Remove least important memories when capacity is reached.
        """
        # Calculate current scores
        scores = []
        current_time = datetime.now()
        
        for memory in self.memories:
            # Time decay
            time_diff = (current_time - memory.timestamp).total_seconds() / 86400  # days
            time_factor = np.exp(-self.time_decay_factor * time_diff)
            
            # Combined score
            score = memory.importance * time_factor * (1 + memory.access_count * 0.1)
            scores.append(score)
            
        # Keep only top memories
        keep_indices = np.argsort(scores)[-self.capacity:]
        self.memories = [self.memories[i] for i in keep_indices]

    def clear(self) -> None:
        """
        Clear all memories.
        """
        self.memories = []

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get memory system statistics.
        """
        if not self.memories:
            return {
                "total_memories": 0,
                "average_importance": 0,
                "average_access_count": 0
            }
            
        return {
            "total_memories": len(self.memories),
            "average_importance": np.mean([m.importance for m in self.memories]),
            "average_access_count": np.mean([m.access_count for m in self.memories])
        }

    def state_dict(self) -> Dict:
        """
        Get memory system state for saving.
        """
        return {
            "memories": self.memories,
            "capacity": self.capacity,
            "importance_threshold": self.importance_threshold,
            "time_decay_factor": self.time_decay_factor,
            "access_boost": self.access_boost
        }

    def load_state_dict(self, state: Dict) -> None:
        """
        Load memory system state.
        """
        self.memories = state["memories"]
        self.capacity = state["capacity"]
        self.importance_threshold = state["importance_threshold"]
        self.time_decay_factor = state["time_decay_factor"]
        self.access_boost = state["access_boost"]
