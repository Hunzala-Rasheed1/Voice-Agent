# Voice AI Agent

A low-latency voice assistant that combines Speech-to-Text (STT), Language Model (LLM), and Text-to-Speech (TTS) capabilities to create a responsive conversational AI experience.

## Features

- **Fast Response Times**: Optimized for minimal latency with response caching
- **Dual Interface**: Supports both voice-based and text-based interactions
- **Automatic Environment Detection**: Adapts to Colab or local environment
- **Performance Tracking**: Monitors and reports latency metrics
- **Optimized Resource Usage**: Configurable threading and memory usage

## Requirements

### Python Dependencies
See `requirements.txt` for a complete list of Python dependencies.

### External Dependencies
- **FFmpeg**: Optional but recommended for audio conversion (improves TTS quality)
- **Microphone**: Required for voice-based conversation mode

### Model Files
- **Sherpa-ONNX Models**: Required for speech recognition
  - The code assumes Sherpa-ONNX Zipformer English models in a specific directory
  - Default path: `G://Voice_Assistence//sherpa-onnx-zipformer-en-2023-04-01//`

## Installation

1. Clone this repository or download the script
2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Install FFmpeg (optional but recommended):
   - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
   - **Linux**: `sudo apt install ffmpeg`
   - **macOS**: `brew install ffmpeg`
4. Download Sherpa-ONNX models:
   - Get the English Zipformer models from [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
   - Update the model paths in the configuration section of the script

## Configuration

The assistant is configured through a Python dictionary in the script. Key configuration options:

- **STT**: Paths to Sherpa ONNX model files (encoder, decoder, joiner, tokens)
- **LLM**: Model name and common response templates
- **TTS**: Common phrases to pre-cache for faster response

Example configuration structure:
```python
config = {
    "stt": {
        "encoder": "path/to/encoder.onnx",
        "decoder": "path/to/decoder.onnx",
        "joiner": "path/to/joiner.onnx",
        "tokens": "path/to/tokens.txt"
    },
    "llm": {
        "model_name": "gpt2",  # Or another Hugging Face model
        "common_responses": {
            "hello": "Hi there! How can I help you?",
            # Add more common patterns here
        }
    },
    "tts": {
        "common_phrases": [
            "Hi there! How can I help you?",
            # Add more common phrases here
        ]
    }
}
```

## Usage

Run the script with Python:

```
python voice_ai_agent.py
```

The assistant will:
1. Initialize all components (STT, LLM, TTS)
2. Detect your environment (Colab vs local)
3. Check for microphone availability
4. Start either voice-based or text-based conversation mode

### Voice Conversation Mode
- Speak into your microphone
- The assistant will transcribe your speech, generate a response, and speak back
- Say "quit", "exit", or "bye" to end the conversation

### Text Conversation Mode
- Type your message and press Enter
- The assistant will generate a text response and save the audio to a WAV file
- Type "quit", "exit", or "bye" to end the conversation

## Performance Optimization

To improve performance:
- Use a GPU if available (the code automatically detects and uses CUDA)
- Increase the cache of common responses and phrases
- Adjust the number of threads used by Sherpa STT
- Use a smaller LLM model for faster response times

## Limitations

- The demo uses gTTS which requires internet connectivity
- Voice mode requires a working microphone and PyAudio
- Model files need to be downloaded separately
- Path configuration may need adjustment for your environment
