from typing import Dict, List, Optional, Tuple
import numpy as np
from midiutil import MIDIFile
import pygame.midi
import librosa
from ..utils.music_theory import (
    SCALES,
    CHORD_PROGRESSIONS,
    GENRE_PATTERNS
)

class MusicGenerator:
    """
    Music generation module that creates melodies, harmonies, and complete
    musical arrangements.
    """
    
    def __init__(self):
        self.sample_rate = 44100
        self.tempo = 120  # BPM
        pygame.midi.init()
        
        # Musical parameters
        self.scales = SCALES
        self.chord_progressions = CHORD_PROGRESSIONS
        self.genre_patterns = GENRE_PATTERNS

    def compose(self, 
               genre: str = "electronic", 
               mood: str = "uplifting",
               duration: int = 60,
               key: str = "C",
               scale: str = "major") -> bytes:
        """
        Compose a complete musical piece.
        
        Args:
            genre (str): Musical genre
            mood (str): Emotional mood of the piece
            duration (int): Duration in seconds
            key (str): Musical key
            scale (str): Musical scale
            
        Returns:
            bytes: MIDI file content
        """
        # Create MIDI file
        midi = MIDIFile(4)  # 4 tracks: melody, harmony, bass, drums
        
        # Set tempo
        midi.addTempo(0, 0, self.tempo)
        
        # Generate components
        melody = self._generate_melody(key, scale, duration)
        harmony = self._generate_harmony(melody, key)
        bass = self._generate_bass_line(harmony)
        drums = self._generate_drums(genre, duration)
        
        # Add tracks to MIDI file
        self._add_track(midi, 0, melody, "melody")
        self._add_track(midi, 1, harmony, "harmony")
        self._add_track(midi, 2, bass, "bass")
        self._add_track(midi, 3, drums, "drums")
        
        # Convert to bytes
        midi_data = self._midi_to_bytes(midi)
        return midi_data

    def _generate_melody(self, key: str, scale: str, duration: int) -> List[Tuple[int, int, int]]:
        """
        Generate a melodic line.
        Returns: List of (note, duration, velocity) tuples
        """
        scale_notes = self.scales[scale]
        base_note = self._note_to_number(key)
        notes = []
        
        time = 0
        while time < duration:
            # Choose note length (quarter, eighth, etc.)
            note_length = np.random.choice([0.25, 0.5, 1.0], p=[0.3, 0.5, 0.2])
            
            # Choose note from scale
            note = base_note + np.random.choice(scale_notes)
            velocity = np.random.randint(60, 100)
            
            notes.append((note, note_length, velocity))
            time += note_length
            
        return notes

    def _generate_harmony(self, melody: List[Tuple[int, int, int]], key: str) -> List[Tuple[List[int], int, int]]:
        """
        Generate harmonic progression based on melody.
        Returns: List of (chord_notes, duration, velocity) tuples
        """
        progression = self.chord_progressions[np.random.choice(list(self.chord_progressions.keys()))]
        base_note = self._note_to_number(key)
        chords = []
        
        for chord_type in progression:
            chord_notes = [base_note + note for note in chord_type]
            duration = 4.0  # One bar
            velocity = 70
            chords.append((chord_notes, duration, velocity))
            
        return chords

    def _generate_bass_line(self, harmony: List[Tuple[List[int], int, int]]) -> List[Tuple[int, int, int]]:
        """
        Generate bass line based on harmony.
        Returns: List of (note, duration, velocity) tuples
        """
        bass_line = []
        
        for chord, duration, _ in harmony:
            # Use root note of each chord
            root_note = chord[0] - 12  # One octave down
            bass_line.append((root_note, duration, 80))
            
        return bass_line

    def _generate_drums(self, genre: str, duration: int) -> List[Tuple[int, int, int]]:
        """
        Generate drum pattern based on genre.
        Returns: List of (drum_note, duration, velocity) tuples
        """
        pattern = self.genre_patterns.get(genre, self.genre_patterns["default"])
        drums = []
        
        time = 0
        while time < duration:
            for drum_hit in pattern:
                note, length, velocity = drum_hit
                drums.append((note, length, velocity))
                time += length
                
        return drums

    def _add_track(self, midi: MIDIFile, track: int, notes: List[Tuple], track_type: str):
        """
        Add a track to the MIDI file.
        """
        time = 0
        for note_data in notes:
            if track_type == "harmony":
                # Chord
                chord_notes, duration, velocity = note_data
                for note in chord_notes:
                    midi.addNote(track, 0, note, time, duration, velocity)
            else:
                # Single note
                note, duration, velocity = note_data
                midi.addNote(track, 0, note, time, duration, velocity)
            time += duration

    def _note_to_number(self, note: str) -> int:
        """
        Convert note name to MIDI note number.
        """
        notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        octave = 4  # Middle octave
        note_idx = notes.index(note.upper())
        return note_idx + (octave * 12) + 60

    def _midi_to_bytes(self, midi: MIDIFile) -> bytes:
        """
        Convert MIDI file to bytes.
        """
        import io
        buffer = io.BytesIO()
        midi.writeFile(buffer)
        return buffer.getvalue()

    def export_to_file(self, midi_data: bytes, filename: str):
        """
        Export MIDI data to a file.
        """
        with open(filename, 'wb') as f:
            f.write(midi_data)
