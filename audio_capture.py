# Core Imports
import os
import time
import wave
import sys
import logging
import ssl
import certifi
from datetime import datetime
from threading import Thread
from typing import Optional, Set, List

# Third-party Imports
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import pyperclip
from pynput import keyboard
import rumps

# Local Imports
from model_manager import ModelManager

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SSL Configuration
ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl_context.verify_mode = ssl.CERT_REQUIRED

# Audio Configuration Constants
SAMPLE_RATE = 44100
CHANNELS = 1
BLOCK_SIZE = 8192
AUDIO_DTYPE = np.int16

class AudioNotifier:
    """Handle system sound notifications."""
    
    SOUNDS = {
        'start': '/System/Library/Sounds/Pop.aiff',
        'stop': '/System/Library/Sounds/Bottle.aiff',
        'success': '/System/Library/Sounds/Glass.aiff',
        'error': '/System/Library/Sounds/Basso.aiff'
    }
    
    @staticmethod
    def play_sound(sound_type: str) -> None:
        """Play a system notification sound."""
        try:
            if sound_type in AudioNotifier.SOUNDS:
                sound_file = AudioNotifier.SOUNDS[sound_type]
                if os.path.exists(sound_file):
                    os.system(f'afplay {sound_file} &')
        except Exception as e:
            logger.error(f"Error playing sound: {e}")

class AudioTranscriberApp(rumps.App):
    """Main application class for the audio transcriber."""
    
    def __init__(self):
        super().__init__(
            "Audio Transcriber",
            title="🎤",
            quit_button=None
        )
        
        self.setup_processor()
        self.setup_menu()
        self.setup_icon_refresh()
        logger.info("Audio Transcriber running in background")

    def setup_processor(self):
        self.processor = AudioProcessor(self)

    def setup_menu(self):
        self.menu = [
            rumps.MenuItem("Start/Stop Recording (⌘+⇧+9)", callback=self.toggle_recording),
            None,
            rumps.MenuItem("Quit", callback=self.quit_app)
        ]

    def setup_icon_refresh(self):
        rumps.Timer(self.refresh_icon, 60).start()

    def refresh_icon(self, _):
        if not self.processor.is_recording:
            current_title = self.title
            self.title = current_title

    def quit_app(self, _):
        logger.info("Quitting application")
        self.processor.cleanup()
        
        try:
            from run_transcriber import cleanup_logs
            cleanup_logs()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        rumps.quit_application()

    def stop(self):
        self.processor.cleanup()

    def toggle_recording(self, _):
        self.processor.toggle_recording()

class AudioProcessor:
    """Handle audio recording and processing."""
    
    def __init__(self, app):
        self.initialize_attributes(app)
        self.setup_keyboard_listener()
        self.setup_icon_refresh()

    def initialize_attributes(self, app):
        self.app = app
        
        # Audio settings
        self.sample_rate = SAMPLE_RATE
        self.channels = CHANNELS
        self.dtype = AUDIO_DTYPE
        self.blocksize = BLOCK_SIZE
        
        # State management
        self.is_recording = False
        self.ready_to_record = True
        self.frames = []
        
        # Components
        self.model_manager = ModelManager()
        self._icon_state = "🎤"
        self.keys_pressed = set()

    def setup_keyboard_listener(self):
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()

    def setup_icon_refresh(self):
        rumps.Timer(self.refresh_icon_state, 1).start()

    @property
    def icon_state(self):
        return self._icon_state

    @icon_state.setter
    def icon_state(self, value):
        self._icon_state = value
        try:
            self.app.title = value
        except Exception as e:
            logger.error(f"Error updating icon: {e}")

    def toggle_recording(self) -> None:
        if not self.is_recording and self.ready_to_record:
            logger.info("Starting new recording")
            Thread(target=self.start_recording).start()
        elif self.is_recording:
            logger.info("Stopping recording")
            self.stop_recording()

    def start_recording(self) -> None:
        if not self.is_recording and self.ready_to_record:
            self.icon_state = "⏺️"
            self.frames = []
            self.is_recording = True
            
            try:
                AudioNotifier.play_sound('start')
                with sd.InputStream(
                    callback=self.callback,
                    channels=self.channels,
                    samplerate=self.sample_rate,
                    blocksize=self.blocksize,
                    dtype=np.float32
                ):
                    while self.is_recording:
                        time.sleep(0.1)
                        
            except Exception as e:
                logger.error(f"Error during recording: {e}")
                self.is_recording = False
                self.icon_state = "🎤"
                AudioNotifier.play_sound('error')

    def stop_recording(self) -> None:
        self.is_recording = False
        AudioNotifier.play_sound('stop')
        
        if self.frames:
            self.process_recorded_audio()

    def callback(self, indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        audio_data = (indata * np.iinfo(np.int16).max).astype(np.int16)
        self.frames.append(audio_data.copy())

    def process_recorded_audio(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"recording_{timestamp}.wav"
        
        audio_data = self.save_audio(audio_filename)
        
        if audio_data is not None:
            self.handle_transcription(audio_data, audio_filename)
        else:
            AudioNotifier.play_sound('error')
            
        self.icon_state = "🎤"
        self.ready_to_record = True

    def save_audio(self, filename: str) -> Optional[np.ndarray]:
        if not self.frames:
            return None
            
        try:
            audio_data = np.concatenate(self.frames)
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data.tobytes())
            return audio_data
            
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            return None

    def handle_transcription(self, audio_data: np.ndarray, audio_filename: str):
        try:
            transcript = self.transcribe_audio(audio_data)
            if transcript:
                pyperclip.copy(transcript)
                self.icon_state = "✅"
                AudioNotifier.play_sound('success')
                time.sleep(1)
            else:
                AudioNotifier.play_sound('error')
        finally:
            try:
                os.remove(audio_filename)
            except Exception as e:
                logger.error(f"Could not delete audio file: {e}")
                AudioNotifier.play_sound('error')

    def transcribe_audio(self, audio_data: np.ndarray) -> Optional[str]:
        try:
            self.icon_state = "💭"
            model = self.ensure_model_loaded()
            temp_file = "temp_recording.wav"
            
            return self.perform_transcription(model, audio_data, temp_file)
                    
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return None
        finally:
            self.icon_state = "🎤"

    def perform_transcription(self, model: WhisperModel, audio_data: np.ndarray, temp_file: str) -> Optional[str]:
        self.save_temp_audio(audio_data, temp_file)
        
        try:
            segments, _ = model.transcribe(
                temp_file,
                beam_size=5,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            return self.process_transcription_segments(segments)
                
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def save_temp_audio(self, audio_data: np.ndarray, temp_file: str):
        with wave.open(temp_file, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())

    def process_transcription_segments(self, segments) -> Optional[str]:
        text_segments = []
        for segment in segments:
            segment_text = segment.text.strip()
            if segment_text:
                text_segments.append(segment_text)
        
        if text_segments:
            text = ' '.join(text_segments)
            return self.process_text(text)
        else:
            logger.warning("No speech detected in audio")
            return None

    def process_text(self, text: str) -> str:
        if not text:
            return text
            
        sentences = []
        parts = text.replace('. ', '.').split('.')
        
        for part in parts:
            if not part.strip():
                continue
                
            cleaned = part.strip()
            
            if cleaned:
                cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()
            
            if cleaned and not cleaned.endswith(('!', '?', '.')):
                cleaned += '.'
            
            sentences.append(cleaned)
        
        return ' '.join(sentences)

    def ensure_model_loaded(self) -> WhisperModel:
        try:
            return self.model_manager.get_model()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.icon_state = "❌"
            AudioNotifier.play_sound('error')
            raise

    def refresh_icon_state(self, _):
        if not self.is_recording:
            try:
                current = self.icon_state
                self.icon_state = "🎤 "
                time.sleep(0.1)
                self.icon_state = current
            except Exception as e:
                logger.error(f"Error in icon refresh: {e}")

    def on_press(self, key: keyboard.Key) -> None:
        try:
            if isinstance(key, keyboard.KeyCode):
                if key.char is not None:
                    self.keys_pressed.add(key.char.lower())
            else:
                self.keys_pressed.add(key)
            
            if (keyboard.Key.cmd in self.keys_pressed and 
                keyboard.Key.shift in self.keys_pressed and 
                '9' in self.keys_pressed):
                self.toggle_recording()
        except Exception as e:
            logger.error(f"Error in key press handler: {e}")

    def on_release(self, key: keyboard.Key) -> None:
        try:
            if isinstance(key, keyboard.KeyCode):
                if key.char is not None:
                    self.keys_pressed.discard(key.char.lower())
            elif key is not None:
                self.keys_pressed.discard(key)
        except Exception as e:
            logger.error(f"Error in key release handler: {e}")

    def cleanup(self):
        self.listener.stop()

def main():
    """Main function to run the audio transcriber app."""
    try:
        app = AudioTranscriberApp()
        app.run()
    except Exception as e:
        logger.error(f"Error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()