#!/usr/bin/env python3
"""
Warmup script for Orpheus TTS API.
Sends a couple of requests to warm up the model for optimal performance.
"""
import requests
import time
import os
import logging
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def warmup_api(host="localhost", port=8000, save_audio=False):
    """Send warmup requests to the API"""
    base_url = f"http://{host}:{port}"
    
    # First check if API is up
    try:
        logger.info(f"Checking if API is available at {base_url}...")
        response = requests.get(f"{base_url}/api/voices", timeout=5)
        if response.status_code != 200:
            logger.error(f"API returned status code {response.status_code}")
            return False
        voices = response.json()
        logger.info(f"API is up. Available voices: {[v['name'] for v in voices['voices']]}")
    except requests.RequestException as e:
        logger.error(f"API is not available: {str(e)}")
        logger.info("Make sure the API server is running (./start.sh)")
        return False

    # Warmup requests for each voice
    test_text = "This is a warmup request to optimize the text-to-speech model performance."
    
    for voice in ["female", "male"]:
        logger.info(f"Sending warmup request with voice '{voice}'...")
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/v1/audio/speech/stream",
                json={"input": test_text, "voice": voice},
                stream=True,
                headers={"Accept-Encoding": "identity", "Cache-Control": "no-store"}
            )
            
            # Process the streaming response
            audio_chunks = bytearray()
            first_chunk = True
            
            for chunk in response.iter_content(chunk_size=1):
                if first_chunk:
                    ttfb = time.time() - start_time
                    logger.info(f"Time to first byte: {ttfb*1000:.0f} ms")
                    first_chunk = False
                audio_chunks.extend(chunk)
            
            total_time = time.time() - start_time
            logger.info(f"Voice '{voice}' warmup complete ({len(audio_chunks)} bytes in {total_time:.2f} seconds)")
            
            # Save audio file if requested
            if save_audio:
                os.makedirs("warmup_audio", exist_ok=True)
                output_path = f"warmup_audio/warmup_{voice}.pcm"
                with open(output_path, "wb") as f:
                    f.write(audio_chunks)
                logger.info(f"Saved warmup audio to {output_path}")
                
        except requests.RequestException as e:
            logger.error(f"Error during warmup for voice '{voice}': {str(e)}")
    
    logger.info("Model warmup completed!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Warm up the Orpheus TTS API")
    parser.add_argument("--host", default="localhost", help="API host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="API port (default: 8000)")
    parser.add_argument("--save", action="store_true", help="Save generated audio files")
    
    args = parser.parse_args()
    warmup_api(args.host, args.port, args.save)
