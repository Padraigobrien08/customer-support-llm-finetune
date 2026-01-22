#!/usr/bin/env python3
"""
Minimal model server for the customer-support fine-tuned adapter.
"""

import os
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class GenerateRequest(BaseModel):
    messages: list[Message]
    max_new_tokens: int | None = None


class GenerateResponse(BaseModel):
    content: str


app = FastAPI(title="Customer Support Model API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
tokenizer = None
device = "cpu"
supports_chat_template = False
system_prompt = None
default_max_new_tokens = 250


def format_messages_simple(
    messages: list[dict[str, Any]], system_prompt_text: str | None = None
) -> str:
    parts = []
    if system_prompt_text:
        parts.append(f"System: {system_prompt_text}")
    for msg in messages:
        role_label = msg["role"].capitalize()
        parts.append(f"{role_label}: {msg['content']}")
    parts.append("Assistant:")
    return "\n".join(parts)


def format_messages_with_template(tokenizer_obj, messages: list[dict[str, Any]], system_prompt_text: str | None = None) -> str:
    chat_messages: list[dict[str, str]] = []
    if system_prompt_text:
        chat_messages.append({"role": "system", "content": system_prompt_text})
    chat_messages.extend(messages)
    return tokenizer_obj.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True
    )


def _clean_response(text: str) -> str:
    """
    Clean and truncate the response to ensure it's concise and on-topic.
    """
    if not text:
        return text
    
    # Split into sentences (handle multiple sentence endings)
    import re
    sentences = re.split(r'[.!?]+\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Limit to reasonable number of sentences for customer support (4-5 max)
    max_sentences = 5
    if len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]
        # Rejoin with periods
        text = '. '.join(sentences) + '.'
    
    # Check for repetitive content
    if len(text) > 200:
        # Look for repeated phrases
        words = text.lower().split()
        if len(words) > 50:
            # Check for repeated 10-word sequences
            for i in range(len(words) - 20):
                seq = ' '.join(words[i:i+10])
                # Check if this sequence appears later
                later_text = ' '.join(words[i+10:])
                if seq in later_text:
                    # Found repetition - truncate at the first occurrence
                    word_index = text.lower().find(seq, i + 10)
                    if word_index != -1:
                        # Find the sentence boundary before this point
                        truncate_point = text.rfind('.', 0, word_index)
                        if truncate_point > 50:  # Only truncate if we have enough content
                            text = text[:truncate_point + 1].strip()
                            break
    
    # Remove common rambling patterns
    # Look for phrases that suggest the response is going off-topic
    rambling_indicators = [
        "based on",
        "based",
        "let me connect you",
        "would they like me",
        "nope",
        "maybe yes",
        "you don't need",
        "i guarantee",
        "i never ask",
        "that's private",
        "i wouldn't share",
        "phone number",
        "credit card",
        "social security",
        "are there any other ways",
        "you deserve the best",
    ]
    
    # Find where rambling might start
    text_lower = text.lower()
    for indicator in rambling_indicators:
        idx = text_lower.find(indicator)
        if idx > 200:  # Only if it's later in the response
            # Truncate at the sentence before this indicator
            truncate_point = text.rfind('.', 0, idx)
            if truncate_point > 50:  # Keep at least some content
                text = text[:truncate_point + 1].strip()
                break
    
    # Final length check - customer support responses shouldn't be too long
    max_chars = 500
    if len(text) > max_chars:
        # Check if we're in a numbered list - don't cut off mid-list
        import re
        # Look for numbered list patterns like "1.)", "2.)", "3.)" or "1.", "2.", "3."
        numbered_list_pattern = r'\d+[.)]\s'
        matches = list(re.finditer(numbered_list_pattern, text[:max_chars + 50]))
        
        if matches:
            # Find the last complete numbered item before max_chars
            last_complete_item = None
            for match in reversed(matches):
                item_end = match.end()
                # Find the end of this item (next number or end of text)
                next_match = None
                for next_match_obj in matches:
                    if next_match_obj.start() > match.start():
                        next_match = next_match_obj
                        break
                
                if next_match:
                    # Check if this item is complete (ends before next item or max_chars)
                    item_text = text[match.start():next_match.start()]
                    if len(item_text) > 10 and match.end() < max_chars:
                        # This item is complete
                        last_complete_item = next_match.start()
                        break
                elif item_end < max_chars:
                    # Last item, check if it's reasonably complete
                    item_text = text[match.start():]
                    if len(item_text) > 10:
                        # Keep the whole last item if it's not too long
                        if len(text[match.start():]) < max_chars + 100:
                            last_complete_item = len(text)
                            break
            
            if last_complete_item and last_complete_item > 100:
                text = text[:last_complete_item].strip()
            else:
                # Fallback: truncate at sentence boundary
                truncate_point = text.rfind('.', 0, max_chars)
                if truncate_point > 50:
                    text = text[:truncate_point + 1].strip()
                else:
                    # Last resort: just truncate
                    text = text[:max_chars].strip()
                    if not text.endswith(('.', '!', '?')):
                        text += '.'
        else:
            # No numbered list, truncate at sentence boundary
            truncate_point = text.rfind('.', 0, max_chars)
            if truncate_point > 50:
                text = text[:truncate_point + 1].strip()
            else:
                # Fallback: just truncate
                text = text[:max_chars].strip()
                if not text.endswith(('.', '!', '?')):
                    text += '.'
    
    return text.strip()


@app.on_event("startup")
def load_model() -> None:
    global model, tokenizer, device, supports_chat_template, system_prompt, default_max_new_tokens

    model_id = os.getenv("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    adapter_dir_env = os.getenv("ADAPTER_DIR", "outputs/smoke_001")
    device_env = os.getenv("DEVICE", "auto")
    default_max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "250"))

    project_root = Path(__file__).parent.parent
    adapter_dir = Path(adapter_dir_env)
    if not adapter_dir.is_absolute():
        adapter_dir = project_root / adapter_dir

    if not adapter_dir.exists():
        raise RuntimeError(f"Adapter directory not found: {adapter_dir}")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("Missing model dependencies. Install from requirements.txt") from exc

    if device_env == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = device_env

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    supports_chat_template = hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None

    model_dtype = torch.float32 if device == "cpu" else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=model_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    if device != "cuda":
        base_model = base_model.to(device)

    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    try:
        from csft.prompts import load_system_prompt

        system_prompt = load_system_prompt(project_root / "prompts" / "system.txt")
    except Exception:
        system_prompt = None


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate_reply(payload: GenerateRequest) -> GenerateResponse:
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        import torch
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="Torch not available") from exc

    messages = [{"role": msg.role, "content": msg.content} for msg in payload.messages]

    if supports_chat_template:
        prompt = format_messages_with_template(tokenizer, messages, system_prompt)
    else:
        prompt = format_messages_simple(messages, system_prompt)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    max_new_tokens = payload.max_new_tokens or default_max_new_tokens

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.85,
            repetition_penalty=1.5,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=4,
            early_stopping=True,
        )

    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    # Post-process to clean up the response
    response_text = _clean_response(response_text)

    return GenerateResponse(content=response_text)
