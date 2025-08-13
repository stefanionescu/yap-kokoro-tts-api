import asyncio
import torch
import os
import logging
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer
import threading
import queue
from src.decoder import tokens_decoder_sync

logger = logging.getLogger(__name__)

class OrpheusModel:
    def __init__(self, model_name="canopylabs/orpheus-3b-0.1-ft", 
                 tokenizer=None, 
                 max_model_len=None, 
                 gpu_memory_utilization=0.9, 
                 max_num_batched_tokens=8192, 
                 max_num_seqs=4, 
                 enable_chunked_prefill=True,
                 quantization="deepspeedfp"):
        self.model_name = model_name
        self.quantization = quantization
        self.available_voices = ["female", "male"]  # API voice options
        self._voice_mapping = {"female": "tara", "male": "zac"}  # Internal mapping to model voices
        # Prefer env overrides to avoid config conflicts
        env_max_len = os.getenv("TRT_MAX_SEQ_LEN") or os.getenv("MAX_MODEL_LEN")
        self.max_model_len = int(env_max_len) if env_max_len else (max_model_len or 8192)
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.enable_chunked_prefill = enable_chunked_prefill
        
        logger.info(f"Initializing OrpheusModel with model={model_name}, quantization={quantization}")
        self.engine = self._setup_engine()
        
        # Use provided tokenizer path or default to model_name
        tokenizer_path = tokenizer if tokenizer else model_name
        self.tokenizer = self._load_tokenizer(tokenizer_path)
        logger.info("OrpheusModel initialization complete")

    def _load_tokenizer(self, tokenizer_path):
        """Load tokenizer from local path or HuggingFace hub with optional auth."""
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        try:
            auth_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
            if os.path.isdir(tokenizer_path):
                return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, use_auth_token=auth_token)
            else:
                return AutoTokenizer.from_pretrained(tokenizer_path, use_auth_token=auth_token)
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
        
    def _setup_engine(self):
        logger.info("Setting up vLLM engine")
        try:
            # Extended context window to match the user's requirements
            engine_args = AsyncEngineArgs(
                model=self.model_name,
                quantization=self.quantization,  # Using DeepSpeed FP6/FP8 quantization
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_num_batched_tokens=self.max_num_batched_tokens,
                max_num_seqs=self.max_num_seqs,
                enable_chunked_prefill=self.enable_chunked_prefill,
                # Auth for gated HF models
                trust_remote_code=True,
                tokenizer_mode="auto",
                download_dir=os.getenv("HF_HOME"),
                enforce_eager=True,
            )
            logger.info(f"vLLM engine args: model={self.model_name}, quantization={self.quantization}, max_model_len={self.max_model_len}")
            return AsyncLLMEngine.from_engine_args(engine_args)
        except Exception as e:
            msg = str(e)
            # Auto-fallback if quantization unsupported or missing configs
            if self.quantization and (
                "Unknown quantization method" in msg or
                "Cannot find the config file for" in msg
            ):
                logger.warning(
                    f"Quantization '{self.quantization}' failed ('{msg}'). Falling back to non-quantized weights."
                )
                try:
                    engine_args = AsyncEngineArgs(
                        model=self.model_name,
                        quantization=None,
                        max_model_len=self.max_model_len,
                        gpu_memory_utilization=self.gpu_memory_utilization,
                        max_num_batched_tokens=self.max_num_batched_tokens,
                        max_num_seqs=self.max_num_seqs,
                        enable_chunked_prefill=self.enable_chunked_prefill,
                        trust_remote_code=True,
                        tokenizer_mode="auto",
                        download_dir=os.getenv("HF_HOME"),
                        enforce_eager=True,
                    )
                    logger.info("Retrying vLLM init without quantization")
                    return AsyncLLMEngine.from_engine_args(engine_args)
                except Exception:
                    logger.exception("Fallback init without quantization failed")
                    raise
            logger.error(f"Failed to initialize vLLM engine: {e}")
            raise
    
    def validate_voice(self, voice):
        if voice:
            if voice not in self.available_voices:
                logger.warning(f"Invalid voice: {voice}. Valid options are: {', '.join(self.available_voices)}")
                raise ValueError(f"Voice {voice} is not available. Valid options are: {', '.join(self.available_voices)}")
    
    def _format_prompt(self, prompt, voice="female"):
        # Map external voice name (female/male) to internal model voice name (tara/zac)
        model_voice = self._voice_mapping.get(voice, "tara")
        logger.debug(f"Formatting prompt with voice: {voice} (maps to model voice: {model_voice})")
        adapted_prompt = f"{model_voice}: {prompt}"
        prompt_tokens = self.tokenizer(adapted_prompt, return_tensors="pt")
        start_token = torch.tensor([[ 128259]], dtype=torch.int64)
        end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
        all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
        prompt_string = self.tokenizer.decode(all_input_ids[0])
        return prompt_string


    def generate_tokens_sync(self, prompt, voice=None, request_id=None, temperature=None, top_p=None, max_tokens=49152, stop_token_ids=[128258], repetition_penalty=None, num_ctx=8192, num_predict=49152):
        """Generate tokens synchronously from the model
        
        Args:
            prompt: The text prompt to convert to speech
            voice: Voice to use ('female' or 'male')
            request_id: Unique identifier for the request (if None, one will be generated)
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Nucleus sampling parameter
            max_tokens: Maximum number of tokens to generate (default: 49152 to match Ollama)
            stop_token_ids: Token IDs to stop generation
            repetition_penalty: Penalty for repeating tokens
            num_ctx: Context window size (default: 8192)
            num_predict: Maximum number of tokens to predict (default: 49152)
            
        Returns:
            Generator yielding tokens
        """
        # Generate a unique request ID if none provided
        if request_id is None:
            import uuid
            request_id = f"req-{str(uuid.uuid4())[:8]}"
        if voice not in self.available_voices:
            logger.warning(f"Invalid voice: {voice}, defaulting to female")
            voice = "female"
            
        # Map external voice to internal model voice
        model_voice = self._voice_mapping.get(voice, "tara")
        
        # Voice-specific parameters as specified by user
        if voice == "female":
            if temperature is None:
                temperature = 0.80
            if top_p is None:
                top_p = 0.80
            if repetition_penalty is None:
                repetition_penalty = 1.90
        else:  # "male" voice
            if temperature is None:
                temperature = 0.4
            if top_p is None:
                top_p = 0.80
            if repetition_penalty is None:
                repetition_penalty = 1.85
            
        logger.info(f"Generating speech for prompt (length: {len(prompt)}), voice: {voice}")
        logger.info(f"Parameters: temp={temperature}, top_p={top_p}, rep_penalty={repetition_penalty}, num_ctx={num_ctx}, num_predict={num_predict}")
        prompt_string = self._format_prompt(prompt, voice)
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids,
            repetition_penalty=repetition_penalty,
        )

        token_queue = queue.Queue()

        async def async_producer():
            try:
                async for result in self.engine.generate(prompt=prompt_string, sampling_params=sampling_params, request_id=request_id):
                    token_queue.put(result.outputs[0].text)
                token_queue.put(None)  # Sentinel to indicate completion
            except Exception as e:
                logger.error(f"Error in token generation: {str(e)}")
                token_queue.put(None)  # Ensure we don't hang

        def run_async():
            asyncio.run(async_producer())

        thread = threading.Thread(target=run_async)
        thread.start()

        try:
            token_count = 0
            while True:
                token = token_queue.get()
                if token is None:
                    break
                token_count += 1
                yield token
                
            logger.info(f"Generated {token_count} tokens for request {request_id}")
        except Exception as e:
            logger.error(f"Error while yielding tokens: {str(e)}")
        finally:
            thread.join()
    
    def generate_speech(self, **kwargs):
        """
        Generate speech audio from text
        
        Returns:
            Generator yielding audio chunks
        """
        logger.debug("Starting speech generation")
        return tokens_decoder_sync(self.generate_tokens_sync(**kwargs))
        
    async def generate_speech_async(self, prompt, voice=None, **kwargs):
        """
        Generate speech audio asynchronously
        
        Args:
            prompt: Text to convert to speech
            voice: Voice to use ('female' or 'male')
            **kwargs: Additional parameters for token generation:
                - temperature: Controls randomness (default depends on voice)
                - top_p: Nucleus sampling parameter (default: 0.8)
                - repetition_penalty: Penalty for repeating tokens (default depends on voice) 
                - num_ctx: Context window size (default: 8192)
                - num_predict: Maximum number of tokens to predict (default: 49152)
                
        Returns:
            Async generator yielding audio chunks
        """
        logger.debug(f"Starting async speech generation for prompt: '{prompt[:20]}...'")
        
        # These match parameters in the Ollama implementation
        kwargs.setdefault('max_tokens', 49152)   # Default to match Ollama's num_predict
        kwargs.setdefault('num_ctx', 8192)       # Default context size
        kwargs.setdefault('num_predict', 49152)  # Default prediction limit
        
        # Extra tokens to process after seeing the end-of-text marker
        # This matches the N_EXTRA_AFTER_EOT value in the Ollama implementation
        N_EXTRA_AFTER_EOT = 8192
        logger.debug(f"Using N_EXTRA_AFTER_EOT={N_EXTRA_AFTER_EOT} to ensure all frames are processed")
        
        token_gen = self.generate_tokens_sync(prompt=prompt, voice=voice, **kwargs)
        
        # Create a queue for audio chunks
        audio_queue = asyncio.Queue()
        
        # Create a task to process tokens and put audio chunks in the queue
        async def process_tokens():
            try:
                frame_buffer = []  # Buffer to collect tokens into frames of 7
                for token in token_gen:
                    frame_buffer.append(token)
                    
                    # Process in groups of 7 (SNAC frame size)
                    if len(frame_buffer) >= 7:
                        audio = await asyncio.to_thread(lambda: next(tokens_decoder_sync(frame_buffer)))
                        frame_buffer = []  # Clear buffer after processing
                        if audio:
                            await audio_queue.put(audio)
                
                # Process any remaining tokens
                if frame_buffer:
                    audio = await asyncio.to_thread(lambda: next(tokens_decoder_sync(frame_buffer)))
                    if audio:
                        await audio_queue.put(audio)
                
                # Signal the end of processing
                await audio_queue.put(None)
            except Exception as e:
                logger.error(f"Error processing tokens: {str(e)}")
                await audio_queue.put(None)
                
        # Start the processing task
        task = asyncio.create_task(process_tokens())
        
        # Yield audio chunks as they become available
        try:
            while True:
                audio_chunk = await audio_queue.get()
                if audio_chunk is None:
                    break
                yield audio_chunk
        finally:
            # Ensure the task is properly awaited
            await task