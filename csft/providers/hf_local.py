"""
HuggingFace local inference provider.

Supports loading and running models locally using transformers.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from csft.providers.base import Provider, ProviderError
from csft.types import ChatMessage, ModelResponse


class HFLocalProvider(Provider):
    """
    HuggingFace local model provider.
    
    Loads models locally using transformers and generates responses.
    Supports chat templates and automatic device detection.
    """
    
    def __init__(
        self,
        model_id: str,
        device: str | None = None,
        dtype: str | None = None,
        max_new_tokens: int = 128,
        adapter_path: str | None = None
    ):
        """
        Initialize HuggingFace local provider.
        
        Args:
            model_id: HuggingFace model identifier (e.g., "meta-llama/Llama-2-7b-chat-hf")
            device: Device to use ("cuda", "cpu", "mps", or None for auto-detection)
            dtype: Data type for model weights ("float16", "bfloat16", "float32", or None for auto)
            max_new_tokens: Maximum number of tokens to generate
            adapter_path: Optional path to a PEFT LoRA adapter directory
            
        Raises:
            ProviderError: If model or tokenizer fails to load
        """
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.adapter_path = adapter_path
        
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        
        # Auto-detect dtype if not specified
        if dtype is None:
            if device == "cuda":
                # Prefer bfloat16 for modern GPUs, fall back to float16
                dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
            else:
                dtype = "float32"
        self.dtype = dtype
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            # Ensure pad_token is set (use eos_token if pad_token is None)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            # Ensure pad_token_id is set
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        except Exception as e:
            raise ProviderError(
                f"Failed to load tokenizer for model '{model_id}': {str(e)}"
            ) from e
        
        # Load model
        try:
            torch_dtype = getattr(torch, dtype)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            if device == "cpu" or device == "mps":
                self.model = self.model.to(device)
            self.model.eval()  # Set to evaluation mode
        except Exception as e:
            raise ProviderError(
                f"Failed to load model '{model_id}': {str(e)}"
            ) from e
        
        # Load LoRA adapter if provided
        if adapter_path:
            try:
                from peft import PeftModel
                from pathlib import Path
                
                adapter_full_path = Path(adapter_path)
                if not adapter_full_path.exists():
                    raise ProviderError(f"Adapter directory not found: {adapter_full_path}")
                
                print(f"Loading LoRA adapter from: {adapter_full_path}", flush=True)
                self.model = PeftModel.from_pretrained(self.model, str(adapter_full_path))
                self.model.eval()  # Ensure eval mode after loading adapter
                print("✓ Adapter loaded", flush=True)
            except ImportError:
                raise ProviderError(
                    "PEFT is required to load adapters. Install with: pip install peft"
                )
            except Exception as e:
                raise ProviderError(f"Failed to load adapter: {str(e)}") from e
        
        # Check if tokenizer supports chat template
        self.supports_chat_template = hasattr(self.tokenizer, "apply_chat_template") and \
                                      self.tokenizer.chat_template is not None
    
    def _format_messages_simple(
        self,
        messages: list[ChatMessage],
        system_prompt: str | None = None
    ) -> str:
        """
        Format messages using a simple template.
        
        Used as fallback when chat template is not available.
        
        Args:
            messages: List of conversation messages
            system_prompt: Optional system prompt
            
        Returns:
            Formatted prompt string
        """
        parts = []
        
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        
        for msg in messages:
            role_label = msg.role.value.capitalize()
            parts.append(f"{role_label}: {msg.content}")
        
        # Add assistant prefix for response
        parts.append("Assistant:")
        
        return "\n".join(parts)
    
    def _format_messages_with_template(
        self,
        messages: list[ChatMessage],
        system_prompt: str | None = None
    ) -> str:
        """
        Format messages using tokenizer's chat template.
        
        Args:
            messages: List of conversation messages
            system_prompt: Optional system prompt
            
        Returns:
            Formatted prompt string
        """
        # Convert to format expected by chat template
        chat_messages = []
        
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        
        for msg in messages:
            chat_messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        
        # Apply chat template
        return self.tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    def generate(
        self,
        messages: list[ChatMessage],
        system_prompt: str | None = None,
        debug: bool = False,
        **kwargs
    ) -> ModelResponse:
        """
        Generate a response from the model.
        
        Args:
            messages: List of conversation messages (excluding system prompt)
            system_prompt: Optional system prompt to prepend
            debug: If True, include detailed debug information in metadata
            **kwargs: Additional generation parameters (temperature, max_new_tokens, etc.)
            
        Returns:
            ModelResponse with generated content and optional metadata
            
        Raises:
            ProviderError: If generation fails
        """
        if not messages:
            raise ProviderError("Cannot generate response: messages list is empty")
        
        # Format prompt - use apply_chat_template with add_generation_prompt=True if available
        if self.supports_chat_template:
            prompt = self._format_messages_with_template(messages, system_prompt)
        else:
            prompt = self._format_messages_simple(messages, system_prompt)
        
        # Tokenize
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        except Exception as e:
            raise ProviderError(f"Failed to tokenize input: {str(e)}") from e
        
        input_length = inputs["input_ids"].shape[1]
        
        # Generation parameters
        # Ensure pad_token_id is set (use eos_token_id if pad_token_id is None)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        
        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", self.max_new_tokens),
            "do_sample": kwargs.get("do_sample", False),  # Deterministic by default
            "pad_token_id": pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Add temperature if sampling is enabled
        if generation_kwargs.get("do_sample", False):
            generation_kwargs["temperature"] = kwargs.get("temperature", 0.7)
        
        # Generate
        try:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    **generation_kwargs
                )
        except Exception as e:
            raise ProviderError(f"Model generation failed: {str(e)}") from e
        
        # Decode the full output (prompt + generated tokens)
        try:
            decoded_full = self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True
            )
        except Exception as e:
            raise ProviderError(f"Failed to decode full model output: {str(e)}") from e
        
        # Decode the prompt portion to extract only the new text
        try:
            decoded_prompt = self.tokenizer.decode(
                inputs["input_ids"][0],
                skip_special_tokens=True
            )
        except Exception as e:
            raise ProviderError(f"Failed to decode prompt: {str(e)}") from e
        
        # Extract new text by removing prompt from full output
        debug_info = {}
        if decoded_full.startswith(decoded_prompt):
            new_text = decoded_full[len(decoded_prompt):].strip()
            if not new_text:
                # Generated text was empty after removing prompt - use full output as fallback
                new_text = decoded_full.strip()
                debug_info["empty_generation_fallback"] = True
                debug_info["fallback_reason"] = "Generated text was empty after removing prompt, using full decoded output"
        else:
            # Prompt doesn't match start - use full output
            new_text = decoded_full.strip()
            debug_info["prompt_mismatch_fallback"] = True
            debug_info["fallback_reason"] = "Prompt does not match start of decoded output, using full decoded output"
        
        # If still empty, this is an error condition - never silently return empty string
        if not new_text:
            error_msg = (
                "Model generated empty response. "
                f"Input tokens: {input_length}, "
                f"Generated tokens: {len(generated_ids[0]) - input_length}, "
                f"Decoded full length: {len(decoded_full)}"
            )
            if debug:
                error_msg += f", Decoded full (first 200): {decoded_full[:200]}, Decoded prompt (first 200): {decoded_prompt[:200]}"
            raise ProviderError(error_msg)
        
        # Calculate generated token count
        generated_token_count = len(generated_ids[0]) - input_length
        
        # Build metadata
        metadata = {
            "provider": "hf_local",
            "model_id": self.model_id,
            "device": self.device,
            "dtype": self.dtype,
            "num_input_tokens": input_length,
            "num_generated_tokens": generated_token_count,
        }
        
        # Add debug information if requested
        if debug:
            metadata["debug"] = {
                "input_token_length": input_length,
                "generated_token_length": generated_token_count,
                "decoded_full_preview": decoded_full[:200],
                "decoded_prompt_preview": decoded_prompt[:200],
                "new_text_length": len(new_text),
            }
            # Merge any fallback debug info
            metadata["debug"].update(debug_info)
        elif debug_info:
            # Include minimal debug info even if debug=False when fallback occurred
            metadata["debug"] = debug_info
        
        return ModelResponse(
            content=new_text,
            metadata=metadata
        )
    
    def __repr__(self) -> str:
        adapter_info = f", adapter_path='{self.adapter_path}'" if self.adapter_path else ""
        return f"HFLocalProvider(model_id='{self.model_id}', device='{self.device}', dtype='{self.dtype}'{adapter_info})"

