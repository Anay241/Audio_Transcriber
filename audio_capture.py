from typing import Optional, List
import os
import time
import wave
from datetime import datetime
from threading import Thread
import logging
import ssl
import certifi
import pyaudio
import numpy as np
from faster_whisper import WhisperModel
import pyperclip
from pynput import keyboard
import rumps
from model_manager import ModelManager

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure SSL context
ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl_context.verify_mode = ssl.CERT_REQUIRED

# Audio Configuration
CHUNK_SIZE = 8192  
FORMAT = pyaudio.paInt16  
CHANNELS = 1  
RATE = 32000  

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
        """Play a system sound."""
        try:
            if sound_type in AudioNotifier.SOUNDS:
                sound_file = AudioNotifier.SOUNDS[sound_type]
                if os.path.exists(sound_file):
                    os.system(f'afplay {sound_file} &')
        except Exception as e:
            logger.error(f"Error playing sound: {e}")

class AudioTranscriberApp(rumps.App):
    def __init__(self):
        logger.debug("Initializing AudioTranscriberApp")
        super().__init__(
            "Audio Transcriber",     # App name
            title="🎤",             # Menu bar icon
            quit_button=None        # Disable default quit button to prevent accidental quits
        )
        
        # Initialize audio processor
        self.processor = AudioProcessor(self)
        
        # Menu items with separator to ensure clickability
        self.menu = [
            rumps.MenuItem("Start/Stop Recording (⌘+⇧+9)", callback=self.toggle_recording),
            None,  # Separator
            rumps.MenuItem("Quit", callback=self.quit_app)
        ]
        
        # Set up periodic icon refresh with shorter interval for smoother transitions
        self._icon_refresh_timer = rumps.Timer(self.refresh_icon, 1)
        self._icon_refresh_timer.start()
        
        # Track current icon to prevent unnecessary updates
        self._current_icon = "🎤"
        
        logger.info("Audio Transcriber running in background")
        logger.info("Use Command+Shift+9 from any application to start/stop recording")

    def refresh_icon(self, _):
        """Periodically refresh the menu bar icon to prevent visual glitches."""
        try:
            if self._current_icon != self.title:
                self.title = self._current_icon
        except Exception as e:
            logger.error(f"Error refreshing icon: {e}")

    def set_icon(self, icon: str):
        """Safely update the menu bar icon."""
        try:
            self._current_icon = icon
            self.title = icon
        except Exception as e:
            logger.error(f"Error setting icon: {e}")

    def quit_app(self, _):
        """Quit the application."""
        logger.info("Quitting application")
        self.processor.cleanup()  # Clean up resources
        
        # Stop icon refresh timer
        if self._icon_refresh_timer:
            self._icon_refresh_timer.stop()
        
        # Import cleanup function from run_transcriber and clean logs
        try:
            from run_transcriber import cleanup_logs
            cleanup_logs()
            logger.info("Application shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        rumps.quit_application()  # Quit the application
    
    def stop(self):
        """Called by rumps when quitting the application."""
        logger.debug("Stopping application")
        self.processor.cleanup()

    def toggle_recording(self, _):
        """Toggle recording state via menu bar."""
        logger.debug("Menu item clicked: toggle recording")
        self.processor.toggle_recording()

class AudioProcessor:
    def __init__(self, app):
        self.app = app
        self.audio = pyaudio.PyAudio()
        
        # Audio settings
        self.format = FORMAT
        self.channels = CHANNELS
        self.rate = RATE
        self.chunk = CHUNK_SIZE
        
        self.is_recording = False
        self.ready_to_record = True
        self.frames = []
        self.stream = None
        self.model_manager = ModelManager()
        
        self.keys_pressed = set()
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.start()
        logger.debug("Keyboard listener started")

    @property
    def icon_state(self):
        return self.app._current_icon

    @icon_state.setter
    def icon_state(self, value):
        self.app.set_icon(value)

    def start_recording(self):
        try:
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
                stream_callback=self._audio_callback,
                input_device_index=None
            )
            self.frames = []
            self.is_recording = True
            self.stream.start_stream()
            logger.info("Recording started with high-quality settings")
            AudioNotifier.play_sound('start')
            self.icon_state = "⏺️"
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            self.icon_state = "❌"
            AudioNotifier.play_sound('error')

    def _audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            self.frames.append(np.frombuffer(in_data, dtype=np.int16))
        return (in_data, pyaudio.paContinue)

    def stop_recording(self):
        if not self.is_recording:
            return

        try:
            self.is_recording = False
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None

            audio_data = np.concatenate(self.frames, axis=0)
            AudioNotifier.play_sound('stop')
            self.icon_state = "💭"
            self._save_and_transcribe(audio_data)
            
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            self.icon_state = "❌"
            AudioNotifier.play_sound('error')
        finally:
            self.frames = []

    def _save_and_transcribe(self, audio_data):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.wav"
            
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)
                wf.setframerate(self.rate)
                wf.writeframes(audio_data.tobytes())
            
            Thread(target=self._transcribe_and_cleanup, 
                  args=(audio_data, filename)).start()
            
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            self.icon_state = "❌"
            AudioNotifier.play_sound('error')
            raise

    def _transcribe_and_cleanup(self, audio_data: np.ndarray, filename: str) -> None:
        try:
            transcription = self.transcribe_audio(audio_data)
            
            if transcription:
                pyperclip.copy(transcription)
                logger.info("Transcription copied to clipboard")
                self.icon_state = "✅"
                time.sleep(1)
                AudioNotifier.play_sound('success')
            else:
                logger.warning("No transcription generated")
                self.icon_state = "❌"
                AudioNotifier.play_sound('error')
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            self.icon_state = "❌"
            AudioNotifier.play_sound('error')
            time.sleep(1)
        finally:
            self.icon_state = "🎤"
            
            max_retries = 3
            retry_delay = 0.5
            
            for attempt in range(max_retries):
                try:
                    if os.path.exists(filename):
                        os.remove(filename)
                        break
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Error removing temporary file after {max_retries} attempts: {e}")
                    else:
                        time.sleep(retry_delay)

    def toggle_recording(self):
        """Toggle between recording and not recording."""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def on_press(self, key):
        """Handle keyboard press events."""
        try:
            self.keys_pressed.add(key)
            # Check for Command+Shift+9
            if (
                keyboard.Key.cmd in self.keys_pressed
                and keyboard.Key.shift in self.keys_pressed
                and hasattr(key, 'char')
                and key.char == '9'
            ):
                self.toggle_recording()
        except Exception as e:
            logger.error(f"Error in keyboard handler: {e}")

    def on_release(self, key):
        """Handle keyboard release events."""
        try:
            self.keys_pressed.discard(key)
        except Exception as e:
            logger.error(f"Error in keyboard handler: {e}")

    def transcribe_audio(self, audio_data: np.ndarray) -> Optional[str]:
        """
        Transcribe audio using Whisper with optimized settings for voice recognition.
        Uses beam search and VAD filtering for better accuracy.
        """
        try:
            logger.info("Starting transcription")
            model = self.model_manager.get_model()
            
            temp_file = "temp_recording.wav"
            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)
                wf.setframerate(self.rate)
                wf.writeframes(audio_data.tobytes())
            
            try:
                segments, _ = model.transcribe(
                    temp_file,
                    beam_size=5,
                    word_timestamps=True,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
                text_segments = []
                for segment in segments:
                    segment_text = segment.text.strip()
                    if segment_text:
                        text_segments.append(segment_text)
                
                if text_segments:
                    return ' '.join(text_segments)
                else:
                    logger.warning("No speech detected in audio")
                    return None
                    
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return None

    def cleanup(self):
        logger.debug("Cleaning up resources")
        
        if self.is_recording:
            self.is_recording = False
        
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.error(f"Error closing audio stream: {e}")
            finally:
                self.stream = None
        
        if self.audio:
            try:
                self.audio.terminate()
            except Exception as e:
                logger.error(f"Error terminating PyAudio: {e}")
            finally:
                self.audio = None
        
        if self.listener:
            try:
                self.listener.stop()
            except Exception as e:
                logger.error(f"Error stopping keyboard listener: {e}")
            finally:
                self.listener = None
        
        self.frames = []
        logger.debug("Cleanup completed")

def main():
    """Main function to run the audio transcriber app."""
    try:
        logger.info("Starting Audio Transcriber")
        app = AudioTranscriberApp()
        app.run()
    except Exception as e:
        logger.error(f"Error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 