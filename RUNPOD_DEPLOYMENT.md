# RunPod Deployment Guide for Orpheus TTS

This guide provides detailed steps for deploying the Orpheus TTS system on RunPod with DeepSpeed FP6/FP8 quantization.

## 1. Access Your RunPod Instance

First, access your RunPod instance via SSH or the web terminal. If you're using SSH:

```bash
ssh <your-pod-username>@<your-pod-ip> -p <port>
```

Or simply open the web terminal from the RunPod dashboard.

## 2. Clone the Repository

Clone your repository onto the RunPod instance:

```bash
# If using GitHub
git clone https://github.com/yourusername/yap-voice-model-deployment.git
cd yap-voice-model-deployment

# OR if uploading directly
# Upload files via SCP or SFTP
# Then navigate to the directory
cd /path/to/yap-voice-model-deployment
```

## 3. Quick Deployment

For a simple one-command deployment, use the provided `run_on_runpod.sh` script:

```bash
chmod +x run_on_runpod.sh
./run_on_runpod.sh
```

This script will:
1. Run the setup script
2. Start the API server in the background
3. Wait for the server to initialize
4. Run the warmup script for optimal performance
5. Display status information and example commands

## 4. Manual Deployment (Step by Step)

If you prefer to run each step manually:

### 4.1. Set Up Environment

```bash
chmod +x setup.sh
./setup.sh
```

When prompted, select option 1 for DeepSpeed FP6/FP8 quantization (recommended for L40S GPUs).

### 4.2. Start the API Server

```bash
chmod +x start.sh
./start.sh
```

Wait until you see output indicating the server is running.

### 4.3. Warm Up the Model (in a separate terminal)

```bash
chmod +x warmup.py
python warmup.py --save
```

This sends test requests to optimize performance.

## 5. Verify the Deployment

Test the API with a simple cURL request:

```bash
# Using female voice
curl -X POST http://localhost:8000/v1/audio/speech/stream \
  -H "Content-Type: application/json" \
  -d '{"input":"This is a test of the Orpheus TTS system.", "voice":"female"}' \
  --output test_female.pcm

# Using male voice
curl -X POST http://localhost:8000/v1/audio/speech/stream \
  -H "Content-Type: application/json" \
  -d '{"input":"This is a test of the Orpheus TTS system.", "voice":"male"}' \
  --output test_male.pcm
```

## 6. Making the API Accessible Externally

To access the API from outside RunPod:

1. Make sure port 8000 is exposed in your RunPod configuration
2. Use your RunPod's public IP address and the exposed port to access the API

Example endpoint: `http://<your-pod-ip>:8000/v1/audio/speech/stream`

**Important:** Ensure you add port 8000 to your RunPod HTTP Ports list in the pod configuration. Port 8888 is typically used for Jupyter Notebook.

## 7. Monitoring and Maintenance

### Check API Server Status

```bash
ps aux | grep uvicorn
```

### View Server Logs

```bash
# If using run_on_runpod.sh
tail -f server.log

# Otherwise check the terminal output where you ran start.sh
```

### Monitor GPU Usage

```bash
nvidia-smi -l 5  # Updates every 5 seconds
```

### Restart the Server

```bash
# Find the process ID
ps aux | grep uvicorn

# Kill the process
kill <process_id>

# Restart
./start.sh
```

## 8. Troubleshooting

### If the model download fails:

1. Check disk space: `df -h`
2. Try manually downloading the model files
3. Set `TRANSFORMERS_OFFLINE=1` in `.env` once the necessary files are present

### If the API server doesn't start:

1. Check for errors in the logs
2. Ensure all dependencies are installed: `pip install -r requirements.txt`
3. Verify CUDA is working: `python -c "import torch; print(torch.cuda.is_available())"`

### If warmup fails:

1. Check if the API server is running
2. Ensure the correct host and port with: `python warmup.py --host <host> --port <port>`
