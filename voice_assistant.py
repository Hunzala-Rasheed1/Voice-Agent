import os
import asyncio
import numpy as np
import time
import torch
import logging
import json
from typing import Dict, List, Optional, Union
import wave
from tempfile import NamedTemporaryFile
import subprocess
import threading
import queue

# Sherpa STT
import sherpa_onnx

# LLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

# TTS
from gtts import gTTS
import io
import soundfile as sf

# Nest asyncio for running in Jupyter
import nest_asyncio
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceAIAgentBackend:
    def __init__(self, config: Dict):
        """Initialize the Voice AI Agent backend with configuration"""
        self.config = config
        
        # Performance tracking
        self.latency_tracker = {
            "stt": [],
            "llm": [],
            "tts": [],
            "total": []
        }
        
        # Initialize components
        self._init_stt()
        self._init_llm()
        self._init_tts()
        
        logger.info("Voice AI Agent Backend initialized")
    
    def _init_stt(self):
        """Initialize Sherpa STT with streaming configuration"""
        logger.info("Initializing Sherpa STT")
        
        # Use from_transducer method
        self.stt = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=self.config["stt"]["encoder"],
            decoder=self.config["stt"]["decoder"],
            joiner=self.config["stt"]["joiner"],
            tokens=self.config["stt"]["tokens"],
            num_threads=4,  # Increased threads for better performance
            decoding_method="greedy_search",
            debug=False
        )
        
        logger.info("Sherpa STT initialized")
    
    def _init_llm(self):
        """Initialize LLM with optimization for minimal latency"""
        logger.info("Initializing LLM")
        
        # Use a small, quantized model for speed
        model_name = self.config["llm"]["model_name"]
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            print("Device set to use GPU")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
            dtype = torch.float16
        else:
            print("Device set to use CPU")
            quantization_config = None
            dtype = torch.float32
        
        # Load the model with updated configuration
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=dtype
        )
        
        self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set up generation config for minimal latency
        self.llm_pipeline = pipeline(
            "text-generation",
            model=self.llm_model,
            tokenizer=self.llm_tokenizer,
            max_new_tokens=50,  # Keep responses short
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.2,
            do_sample=True
        )
        
        # Cache common responses for even faster retrieval
        self.response_cache = {}
        for common_phrase, response in self.config["llm"]["common_responses"].items():
            self.response_cache[common_phrase] = response
            
        logger.info("LLM initialized")
    
    def _init_tts(self):
        """Initialize TTS system (using gTTS for demo)"""
        logger.info("Initializing TTS System")
        
        # For the demo, we're using gTTS
        self.tts_cache = {}
        self.tts_temp_dir = os.path.join(os.getcwd(), "tts_cache")
        os.makedirs(self.tts_temp_dir, exist_ok=True)
        
        # Check if ffmpeg is available
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.ffmpeg_available = True
        except FileNotFoundError:
            logger.warning("FFmpeg not found in PATH. Will use direct audio handling.")
            self.ffmpeg_available = False
        
        # Pre-compute common phrases for minimum latency
        common_phrases = self.config["tts"]["common_phrases"]
        for phrase in common_phrases:
            self._cache_tts_audio(phrase)
            
        logger.info("TTS system initialized")
    
    def _cache_tts_audio(self, text):
        """Pre-cache TTS audio for common phrases"""
        # Create a hash of the text for filename
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cache_path = os.path.join(self.tts_temp_dir, f"{text_hash}.wav")
        
        # Only generate if not already cached
        if not os.path.exists(cache_path):
            # Generate speech
            tts = gTTS(text=text, lang="en", slow=False)
            
            # Save to an in-memory file first (for speed)
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            
            if self.ffmpeg_available:
                # Convert to WAV format with correct parameters for streaming using ffmpeg
                with NamedTemporaryFile(suffix=".mp3", delete=False) as temp_mp3:
                    temp_mp3_path = temp_mp3.name
                    temp_mp3.write(mp3_fp.read())
                    temp_mp3.flush()
                
                try:
                    subprocess.run([ 
                        "ffmpeg", "-y", "-i", temp_mp3_path, 
                        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", 
                        cache_path
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_mp3_path):
                        os.unlink(temp_mp3_path)
            else:
                # Direct save without conversion if ffmpeg is not available
                tts.save(cache_path)
        
        # Store the path in our cache
        self.tts_cache[text] = cache_path
    
    async def generate_response(self, transcript: str) -> str:
        """Generate response using LLM with minimal latency"""
        start_time = time.time()
        
        # Check cache first for immediate response
        if transcript.lower() in self.response_cache:
            response = self.response_cache[transcript.lower()]
            
            # Track latency (near zero for cached responses)
            llm_latency = (time.time() - start_time) * 1000
            self.latency_tracker["llm"].append(llm_latency)
            logger.debug(f"LLM latency (cached): {llm_latency:.2f}ms")
            
            return response
        
        # Prepare prompt - keep it minimal for speed
        prompt = f"User: {transcript}\nAssistant:"
        
        # Generate response with streaming
        response = self.llm_pipeline(
            prompt,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.2,
            return_full_text=False
        )[0]["generated_text"]
        
        # Clean up response
        response = response.strip()
        
        # Track latency
        llm_latency = (time.time() - start_time) * 1000
        self.latency_tracker["llm"].append(llm_latency)
        logger.debug(f"LLM latency: {llm_latency:.2f}ms")
        
        # Cache this response for future use
        if len(self.response_cache) < 1000:  # Limit cache size
            self.response_cache[transcript.lower()] = response
            
        return response
    
    async def synthesize_speech(self, text: str) -> tuple:
        """Synthesize speech with minimal latency"""
        start_time = time.time()
        
        # Check cache first for immediate retrieval
        if text in self.tts_cache:
            # Load the cached audio file
            audio_data, sample_rate = sf.read(self.tts_cache[text])
        else:
            # Not in cache, generate on the fly
            tts = gTTS(text=text, lang="en", slow=False)
            
            # Create a temporary file
            with NamedTemporaryFile(suffix=".mp3", delete=False) as temp_mp3:
                temp_mp3_path = temp_mp3.name
                # Save the MP3 directly to the temp file
                tts.save(temp_mp3_path)
            
            try:
                # Process based on available tools
                if self.ffmpeg_available:
                    # Convert to WAV format with ffmpeg
                    with NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                        temp_wav_path = temp_wav.name
                    
                    subprocess.run([
                        "ffmpeg", "-y", "-i", temp_mp3_path,
                        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", 
                        temp_wav_path
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    # Read the WAV file
                    audio_data, sample_rate = sf.read(temp_wav_path)
                    
                    # Clean up temp WAV file
                    os.unlink(temp_wav_path)
                else:
                    # Read the MP3 directly if ffmpeg is not available
                    import librosa
                    audio_data, sample_rate = librosa.load(temp_mp3_path, sr=16000)
            finally:
                # Clean up temp MP3 file
                if os.path.exists(temp_mp3_path):
                    os.unlink(temp_mp3_path)
            
            # Cache this for future use
            cache_path = os.path.join(self.tts_temp_dir, f"{hash(text)}.wav")
            sf.write(cache_path, audio_data, sample_rate)
            self.tts_cache[text] = cache_path
        
        # Track TTS latency
        tts_latency = (time.time() - start_time) * 1000
        self.latency_tracker["tts"].append(tts_latency)
        logger.debug(f"TTS latency: {tts_latency:.2f}ms")
        
        return audio_data, sample_rate
    
    async def save_audio_response(self, text, output_file):
        """Save the audio response to a file"""
        audio_data, sample_rate = await self.synthesize_speech(text)
        sf.write(output_file, audio_data, sample_rate)
        logger.info(f"Audio response saved to {output_file}")
        return output_file

# Updated configuration
config = {
    "stt": {
        "encoder": "G://Voice_Assistence//sherpa-onnx-zipformer-en-2023-04-01//encoder-epoch-99-avg-1.onnx",
        "decoder": "G://Voice_Assistence//sherpa-onnx-zipformer-en-2023-04-01//decoder-epoch-99-avg-1.onnx",
        "joiner": "G://Voice_Assistence//sherpa-onnx-zipformer-en-2023-04-01//joiner-epoch-99-avg-1.onnx",
        "tokens": "G://Voice_Assistence//sherpa-onnx-zipformer-en-2023-04-01//tokens.txt"
    },
    "llm": {
        "model_name": "gpt2",  # For demo, use smaller models like GPT-2
        "common_responses": {
            "hello": "Hi there! How can I help you?",
            "hi": "Hello! What can I do for you today?",
            "bye": "Goodbye, have a nice day!",
            "what time is it": "I'm sorry, I don't have access to the current time.",
            "what is your name": "I'm your voice assistant, you can call me Claude.",
            "thank you": "You're welcome! Is there anything else I can help with?"
        }
    },
    "tts": {
        "common_phrases": [
            "Hi there! How can I help you?",
            "Hello! What can I do for you today?",
            "Goodbye, have a nice day!",
            "I'm sorry, I don't have access to the current time.",
            "I'm your voice assistant, you can call me Claude.",
            "You're welcome! Is there anything else I can help with?",
            "I didn't catch that. Could you please repeat?",
            "Processing your request..."
        ]
    }
}

# Check if we're in a Colab environment
def is_colab():
    try:
        import google.colab
        return True
    except:
        return False

# Check if microphone is available
def is_microphone_available():
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        input_devices = 0
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info.get('maxInputChannels') > 0:
                input_devices += 1
        p.terminate()
        return input_devices > 0
    except Exception as e:
        print(f"Error checking microphone: {e}")
        return False

# Text-based conversation loop
async def text_conversation(backend):
    print("\n===== Text-based Voice Assistant =====")
    print("Type your message and press Enter. Type 'quit' to exit.")
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            response = "Goodbye! Have a nice day!"
            print(f"\nAssistant: {response}")
            output_file = await backend.save_audio_response(response, "goodbye.wav")
            print(f"Response audio saved to: {output_file}")
            break
        
        # Generate response
        start_time = time.time()
        response = await backend.generate_response(user_input)
        print(f"\nAssistant: {response}")
        
        # Generate audio response
        output_file = await backend.save_audio_response(response, f"response_{int(time.time())}.wav")
        print(f"Response audio saved to: {output_file}")
        
        # Track total latency
        total_latency = (time.time() - start_time) * 1000
        backend.latency_tracker["total"].append(total_latency)
        logger.debug(f"Total latency: {total_latency:.2f}ms")

# Voice-based conversation (with microphone) - for local machines only
async def voice_conversation(backend):
    try:
        import pyaudio
        
        print("\n===== Voice-based Assistant =====")
        print("Speak into your microphone. Say 'quit' to exit.")
        
        # Audio parameters
        sample_rate = 16000
        chunk_size = 1600  # 100ms at 16kHz
        audio_format = pyaudio.paInt16
        channels = 1
        
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        # Open microphone stream
        stream = p.open(
            format=audio_format,
            channels=channels,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_size
        )
        
        print("\nListening... (Press Ctrl+C to stop)")
        
        while True:
            # Create a new STT stream
            stt_stream = backend.stt.create_stream()
            
            # Collect audio until silence
            frames = []
            silence_count = 0
            speaking = False
            
            # Simple Voice Activity Detection
            silence_threshold = 500  # Adjust based on your microphone
            max_silence = 20  # 2 seconds of silence
            
            print("\nSay something...")
            
            try:
                while silence_count < max_silence:
                    # Read audio chunk
                    audio_data = stream.read(chunk_size, exception_on_overflow=False)
                    
                    # Convert to numpy array
                    audio_np = np.frombuffer(audio_data, dtype=np.int16)
                    
                    # Check if speaking (simple energy-based VAD)
                    # Fixed VAD code to prevent warnings
                    audio_squared = audio_np.astype(np.float64)**2
                    mean_squared = np.mean(audio_squared) if len(audio_np) > 0 else 0
                    energy = np.sqrt(max(0, mean_squared))
                    
                    if energy > silence_threshold:
                        silence_count = 0
                        speaking = True
                    elif speaking:
                        silence_count += 1
                        
                    # If speaking detected, process audio
                    if speaking:
                        frames.append(audio_data)
                        
                        # Convert to float32 for Sherpa
                        audio_float = audio_np.astype(np.float32) / 32768.0
                        
                        # Process through Sherpa
                        stt_stream.accept_waveform(sample_rate, audio_float)
                        
                        # Get intermediate results
                        backend.stt.decode_stream(stt_stream)
                        text = stt_stream.result.text
                        
                        if text:
                            print(f"\rTranscript: {text}", end="")
            
            except KeyboardInterrupt:
                print("\nStopped listening.")
                break
            
            # Get final transcript
            if frames:
                # FIXED: Use decode_stream instead of decode_stream_final
                backend.stt.decode_stream(stt_stream)
                transcript = stt_stream.result.text
                
                if transcript:
                    print(f"\nYou said: {transcript}")
                    
                    # Check for quit command
                    if transcript.lower() in ['quit', 'exit', 'bye']:
                        response = "Goodbye! Have a nice day!"
                        print(f"Assistant: {response}")
                        
                        # Synthesize and play response
                        audio_data, audio_sr = await backend.synthesize_speech(response)
                        
                        # Play audio through speakers
                        output_stream = p.open(
                            format=p.get_format_from_width(2),  # 16-bit
                            channels=1,
                            rate=audio_sr,
                            output=True
                        )
                        output_stream.write((audio_data * 32767).astype(np.int16).tobytes())
                        output_stream.stop_stream()
                        output_stream.close()
                        
                        break
                    
                    # Generate response
                    start_time = time.time()
                    response = await backend.generate_response(transcript)
                    print(f"Assistant: {response}")
                    
                    # Synthesize and play response
                    audio_data, audio_sr = await backend.synthesize_speech(response)
                    
                    # Play audio through speakers
                    try:
                        output_stream = p.open(
                            format=p.get_format_from_width(2),  # 16-bit
                            channels=1,
                            rate=audio_sr,
                            output=True
                        )
                        output_stream.write((audio_data * 32767).astype(np.int16).tobytes())
                        output_stream.stop_stream()
                        output_stream.close()
                    except:
                        # If output fails, save to file instead
                        output_file = f"response_{int(time.time())}.wav"
                        sf.write(output_file, audio_data, audio_sr)
                        print(f"Response audio saved to: {output_file}")
                    
                    # Track latency
                    total_latency = (time.time() - start_time) * 1000
                    backend.latency_tracker["total"].append(total_latency)
                    logger.debug(f"Total latency: {total_latency:.2f}ms")
                else:
                    print("\nI didn't hear anything. Please try again.")
            else:
                print("\nNo speech detected. Please try again.")
        
        # Clean up
        stream.stop_stream()
        stream.close()
        p.terminate()
        
    except Exception as e:
        print(f"\nError in voice conversation: {e}")
        print("Falling back to text-based conversation.")
        await text_conversation(backend)
# Main function to run the assistant
async def run_assistant():
    print("Initializing Voice AI Assistant...")
    backend = VoiceAIAgentBackend(config)
    print("Voice AI Assistant initialized!")
    
    # Determine the conversation mode based on environment
    in_colab = is_colab()
    mic_available = is_microphone_available()
    
    if in_colab:
        print("\nRunning in Google Colab environment. Using text-based conversation.")
        await text_conversation(backend)
    elif mic_available:
        print("\nMicrophone detected. Using voice-based conversation.")
        await voice_conversation(backend)
    else:
        print("\nNo microphone detected or PyAudio not installed. Using text-based conversation.")
        await text_conversation(backend)
    
    return backend

# This is needed to run asyncio code in the main thread
if __name__ == "__main__":
    # Run the assistant
    import asyncio
    
    # Create an asyncio event loop
    loop = asyncio.get_event_loop()
    
    try:
        # Run the assistant in the event loop
        backend = loop.run_until_complete(run_assistant())
        
        # Print latency statistics
        if backend.latency_tracker["total"]:
            avg_total = sum(backend.latency_tracker["total"]) / len(backend.latency_tracker["total"])
            print(f"\nAverage total latency: {avg_total:.2f}ms")
        
        # Close the loop
        loop.close()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"\nError running assistant: {e}")