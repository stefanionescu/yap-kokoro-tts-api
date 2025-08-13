import asyncio
import torch
import os
import logging
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer
import threading
import queue
from decoder import tokens_decoder_sync

logger = logging.getLogger(__name__)

class OrpheusModel:
    def __init__(self, model_name="canopylabs/orpheus-3b-0.1-ft", 
                 tokenizer=None, 
                 max_model_len=2048, 
                 gpu_memory_utilization=0.9, 
                 max_num_batched_tokens=8192, 
                 max_num_seqs=4, 
                 enable_chunked_prefill=True,
                 quantization="deepspeedfp"):
        self.model_name = model_name
        self.quantization = quantization
        self.available_voices = ["tara", "zac"]  # Limiting to only needed voices
        self.max_model_len = max_model_len
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
        """Load tokenizer from local path or HuggingFace hub"""
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        try:
            if os.path.isdir(tokenizer_path):
                return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
            else:
                return AutoTokenizer.from_pretrained(tokenizer_path)
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
                # Set extended context window for larger text processing
                max_context_len_to_capture=8192,  # Same as num_ctx in the user's code
            )
            logger.info(f"vLLM engine args: model={self.model_name}, quantization={self.quantization}, max_model_len={self.max_model_len}")
            return AsyncLLMEngine.from_engine_args(engine_args)
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}")
            raise
    
    def validate_voice(self, voice):
        if voice:
            if voice not in self.available_voices:
                logger.warning(f"Invalid voice: {voice}. Valid options are: {', '.join(self.available_voices)}")
                raise ValueError(f"Voice {voice} is not available. Valid options are: {', '.join(self.available_voices)}")
    
    def _format_prompt(self, prompt, voice="tara"):
        logger.debug(f"Formatting prompt with voice: {voice}")
        adapted_prompt = f"{voice}: {prompt}"
        prompt_tokens = self.tokenizer(adapted_prompt, return_tensors="pt")
        start_token = torch.tensor([[ 128259]], dtype=torch.int64)
        end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
        all_input_ids = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
        prompt_string = self.tokenizer.decode(all_input_ids[0])
        return prompt_string


    def generate_tokens_sync(self, prompt, voice=None, request_id="req-001", temperature=None, top_p=None, max_tokens=2000, stop_token_ids=[128258], repetition_penalty=None, num_ctx=8192, num_predict=49152):
        """
        Generate tokens synchronously from the model
        
        Args:
            prompt: The text prompt to convert to speech
            voice: Voice to use (tara for female, zac for male)
            request_id: Unique identifier for the request
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Nucleus sampling parameter
            max_tokens: Maximum number of tokens to generate
            stop_token_ids: Token IDs to stop generation
            repetition_penalty: Penalty for repeating tokens
            num_ctx: Context window size
            num_predict: Maximum number of tokens to predict
            
        Returns:
            Generator yielding tokens
        """
        if voice not in self.available_voices:
            logger.warning(f"Invalid voice: {voice}, defaulting to tara")
            voice = "tara"
        
        # Voice-specific parameters as specified by user
        if voice == "tara":  # Female voice
            if temperature is None:
                temperature = 0.80
            if top_p is None:
                top_p = 0.80
            if repetition_penalty is None:
                repetition_penalty = 1.90
        else:  # "zac" - Male voice
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
            voice: Voice to use
            
        Returns:
            Async generator yielding audio chunks
        """
        logger.debug(f"Starting async speech generation for prompt: '{prompt[:20]}...'")
        
        # Handle N_EXTRA_AFTER_EOT for longer text generation
        kwargs.setdefault('num_predict', 49152)  # Default to higher value for longer texts
        
        # Extra tokens to process after seeing the end-of-text marker
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