# Orpheus TTS Deployment on RunPod with DeepSpeed FP6/FP8

This guide explains how to deploy the Orpheus 3B text-to-speech model on RunPod using vLLM with DeepSpeed FP6/FP8 quantization for maximum performance on L40S GPUs.

## Prerequisites

- A RunPod account with access to GPUs (L40S recommended)
- Basic knowledge of Linux and the command line

## Deployment Steps

### 1. Create a RunPod Instance

1. Log in to your RunPod account
2. Create a new pod with the following specifications:
   - Container image: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
   - GPU: NVIDIA L40S (recommended) or any GPU with at least 24GB VRAM
   - Disk: At least 100GB (the model repository contains unnecessary files that take up space)
   - Ports: Expose port 8000 for the API

### 2. Clone the Repository and Setup

Once your RunPod instance is running, connect to it via SSH or the web terminal and run:

```bash
# Clone the repository
git clone https://github.com/yourusername/yap-voice-model-deployment.git
cd yap-voice-model-deployment

# Run the setup script
./setup.sh
```

The setup script will:
- Install system dependencies
- Create a Python virtual environment
- Install required Python packages
- Set up environment variables in a `.env` file

### 3. Start the API Server

```bash
# Start the API server
./start.sh
```

The server will begin downloading the model from Hugging Face if it's not already cached. This may take some time depending on your internet connection.

> **Note**: The model repository contains optimizer/FSDP files that are not needed for inference but take up a lot of disk space. You can stop the download once the tokenizer and model safetensors files are downloaded, then set `TRANSFORMERS_OFFLINE=1` in the `.env` file and restart the server.

### 4. Optimizing the Model Download

To save disk space and speed up the model loading process:

1. Let the server start downloading the model files
2. Monitor the download progress in the logs
3. Once you see messages indicating the tokenizer and model safetensors files are downloaded, you can stop the server (Ctrl+C)
4. Add `TRANSFORMERS_OFFLINE=1` to your `.env` file
5. Delete unnecessary files from the cache directory:
   ```bash
   find ./cache -name "*.bin" -size +1G -delete
   ```
6. Restart the server with `./start.sh`

## API Endpoints

### Using DeepSpeed FP6/FP8 Quantization

This implementation uses DeepSpeed's FP6/FP8 quantization, which was merged into vLLM in May 2024 (PR #4652). This quantization method provides improved performance on modern GPUs like the L40S compared to previous quantization methods.

Benefits:
- Optimized memory usage
- Faster inference speed
- Better quality-to-performance ratio

The API provides the following endpoints:

### 1. GET `/api/voices`

Returns information about available voices.

**Response:**
```json
{
  "voices": [
    {
      "name": "tara",
      "description": "A natural-sounding female voice.",
      "language": "en",
      "gender": "female",
      "accent": "american"
    },
    {
      "name": "zac",
      "description": "A natural-sounding male voice.",
      "language": "en",
      "gender": "male",
      "accent": "american"
    }
  ],
  "default": "tara",
  "count": 2
}
```

### 2. POST `/v1/audio/speech/stream`

Generates speech audio from text and streams the result as PCM audio.

**Request:**
```json
{
  "input": "Hello, this is a test of the Orpheus TTS system.",
  "voice": "tara"
}
```

**Response:** Audio stream in PCM format.

### 3. WebSocket `/v1/audio/speech/stream/ws`

WebSocket endpoint for streaming speech generation.

**Request message format:**
```json
{
  "input": "Hello, this is a test of the Orpheus TTS system.",
  "voice": "tara",
  "segment_id": "segment1",
  "continue": true
}
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

## Context Parameters for Long-Form Text

For processing longer texts, the system uses:
- Context Window (num_ctx): 8192 tokens
- Max Prediction (num_predict): 49152 tokens
- Extra tokens after end-of-text: 8192 tokens

## Troubleshooting

### Model Download Issues

If you encounter issues downloading the model:

1. Make sure your RunPod has sufficient disk space
2. Try setting `HF_ENDPOINT=https://huggingface.co` in the `.env` file
3. Use `HF_TOKEN=your_huggingface_token` if the model requires authentication

### Memory Issues

If you encounter out-of-memory errors:

1. Reduce the `GPU_MEMORY_UTILIZATION` value in the `.env` file (e.g., to 0.7)
2. Reduce the `MAX_NUM_BATCHED_TOKENS` value
3. Ensure no other GPU-intensive processes are running

## Performance Tuning

For optimal performance on RunPod:

1. Use DeepSpeed FP6/FP8 quantization (`QUANTIZATION=deepspeedfp`) for optimal performance on L40S GPU
2. For processing long texts, ensure NUM_CTX and NUM_PREDICT are set high enough
3. Monitor GPU memory usage with `nvidia-smi` and adjust parameters accordingly