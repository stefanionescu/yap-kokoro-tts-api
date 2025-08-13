# Orpheus TTS Deployment

A FastAPI-based deployment solution for [Canopy Labs' Orpheus 3B](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) text-to-speech model using vLLM with 6-bit quantization.

## Features

- 🚀 **Optimized Performance**: Uses DeepSpeed FP6/FP8 quantization for maximum GPU efficiency
- 🔊 **Low Latency**: Streams audio chunks with low Time-to-First-Byte (TTFB)
- 👥 **Voice Options**: Supports both male (zac) and female (tara) voices with optimized parameters
- 🔌 **API Integration**: RESTful and WebSocket interfaces for easy integration with pipecat
- 📝 **Comprehensive Logging**: Detailed logging for monitoring and debugging

## Quick Start

1. **Setup the environment**:

   ```bash
   ./setup.sh
   ```

2. **Start the server**:

   ```bash
   ./start.sh
   ```

3. **Test the API**:

   ```bash
   curl -X POST http://localhost:8000/v1/audio/speech/stream \
     -H "Content-Type: application/json" \
     -d '{"input":"Hello, this is a test of the Orpheus text-to-speech system.", "voice":"tara"}' \
     --output test.pcm
   ```

## API Documentation

Once the server is running, access the API documentation at:

```
http://localhost:8000/docs
```

## Voice-Specific Settings

The system is configured with optimal parameters for each voice:

### Female Voice (tara)
- Temperature: 0.8
- Top-p: 0.8
- Repetition Penalty: 1.9

### Male Voice (zac)
- Temperature: 0.4  
- Top-p: 0.8
- Repetition Penalty: 1.85

## Context Settings for Long Text

For processing longer texts, the system uses:
- Context Window (num_ctx): 8192 tokens
- Max Prediction (num_predict): 49152 tokens
- Extra tokens after end-of-text: 8192 tokens

## Deployment

For detailed instructions on deploying to RunPod, see [DEPLOYMENT.md](DEPLOYMENT.md).

## Configuration

Configuration is managed through environment variables in the `.env` file. Key settings include:

- `MODEL_NAME`: Path to the Orpheus model
- `QUANTIZATION`: Quantization method (deepspeedfp for FP6/FP8 or awq for alternative 6-bit)
- `GPU_MEMORY_UTILIZATION`: Fraction of GPU memory to use
- `TEMPERATURE_TARA`, `TEMPERATURE_ZAC`, etc.: Voice-specific parameters

## Components

- `main.py`: FastAPI application with API endpoints
- `src/vllm.py`: vLLM integration for optimized model inference
- `src/decoder.py`: SNAC decoder for converting tokens to audio
- `src/logger.py`: Centralized logging setup

## Requirements

- Python 3.10+
- PyTorch 2.1+
- CUDA 11.8+
- vLLM 0.3.3+
- See `requirements.txt` for all dependencies

## Acknowledgements

- [Canopy Labs](https://canopylabs.ai/) for creating the Orpheus TTS model
- [vLLM](https://github.com/vllm-project/vllm) for the efficient LLM inference engine
- [SNAC](https://github.com/hubert-siuzdak/snac/) for the neural audio codec