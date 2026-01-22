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
default_max_new_tokens = 512


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


@app.on_event("startup")
def load_model() -> None:
    global model, tokenizer, device, supports_chat_template, system_prompt, default_max_new_tokens

    model_id = os.getenv("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    adapter_dir_env = os.getenv("ADAPTER_DIR", "outputs/smoke_001")
    device_env = os.getenv("DEVICE", "auto")
    default_max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "512"))

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
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return GenerateResponse(content=response_text)
