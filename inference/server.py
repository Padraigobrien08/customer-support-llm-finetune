#!/usr/bin/env python3
"""
Minimal model server for the customer-support fine-tuned adapter.
"""

import os
import re
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
    temperature: float | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None


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
default_max_new_tokens = 1000  # Increased default - can be overridden by request

# Semantic coherence checker (lazy-loaded)
_semantic_checker = None
_semantic_reference_phrases = [
    "enter your password",
    "click the button",
    "reset your password",
    "check your email",
    "contact support",
    "update your account",
    "verify your identity",
    "access your account",
    "change your settings",
    "track your order",
    "cancel your subscription",
    "request a refund",
    "submit a ticket",
    "speak with a representative",
    "follow these steps",
    "navigate to the settings page",
    "confirm your email address",
    "select an option",
    "enter your information",
    "complete the form",
]


def _check_semantic_coherence(text: str) -> str:
    """
    Check semantic coherence of phrases in the text using embeddings.
    Rewrites incoherent phrases to make sense instead of removing them.
    """
    global _semantic_checker
    
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        # If sentence-transformers not available, skip semantic checking
        return text
    
    # Lazy-load the semantic checker (use a lightweight model)
    if _semantic_checker is None:
        try:
            # Use a small, fast model for semantic similarity
            _semantic_checker = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            # If model loading fails, disable semantic checking
            return text
    
    # Pre-compute reference embeddings once
    try:
        reference_embeddings = _semantic_checker.encode(
            _semantic_reference_phrases, 
            convert_to_numpy=True
        )
    except Exception:
        return text
    
    # Process the text sentence by sentence
    sentences = re.split(r'([.!?]+)', text)
    # Recombine sentences with their punctuation
    sentence_parts = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence_parts.append((sentences[i].strip(), sentences[i + 1]))
        else:
            sentence_parts.append((sentences[i].strip(), ''))
    
    corrected_sentences = []
    
    for sentence, punctuation in sentence_parts:
        if not sentence or len(sentence) < 10:
            if sentence:
                corrected_sentences.append(sentence + punctuation)
            continue
        
        original_sentence = sentence
        words = sentence.split()
        if len(words) < 3:
            corrected_sentences.append(sentence + punctuation)
            continue
        
        # Check phrases of different lengths and find suspicious ones
        phrase_replacements = []  # (original_phrase, corrected_phrase, start_idx, end_idx)
        
        for phrase_len in [4, 5, 6]:  # Focus on longer phrases that are more likely to be weird
            for i in range(len(words) - phrase_len + 1):
                phrase_words = words[i:i+phrase_len]
                phrase = ' '.join(phrase_words).lower()
                
                # Skip if phrase contains URLs/numbers or is too short
                if re.search(r'http|www|https://', phrase, re.IGNORECASE):
                    continue
                
                # Get embedding for this phrase
                try:
                    phrase_embedding = _semantic_checker.encode(phrase, convert_to_numpy=True)
                    
                    # Calculate cosine similarity with all reference phrases
                    similarities = np.dot(reference_embeddings, phrase_embedding) / (
                        np.linalg.norm(reference_embeddings, axis=1) * np.linalg.norm(phrase_embedding)
                    )
                    max_similarity = np.max(similarities)
                    best_match_idx = np.argmax(similarities)
                    
                    # If similarity is very low (< 0.3), the phrase is likely nonsensical
                    if max_similarity < 0.3:
                        # Find the best matching reference phrase
                        best_reference = _semantic_reference_phrases[best_match_idx]
                        
                        # Try to adapt the reference phrase to fit the context
                        # Extract key action words from the reference
                        reference_words = best_reference.split()
                        
                        # If the suspicious phrase has similar structure, try to preserve it
                        # Otherwise, use a simplified version of the reference
                        if len(phrase_words) >= 4:
                            # Try to create a corrected phrase that fits the sentence structure
                            # Use the action from the reference but keep the sentence flow
                            corrected_phrase = _adapt_phrase_to_context(
                                phrase_words, reference_words, words, i
                            )
                        else:
                            corrected_phrase = best_reference
                        
                        phrase_replacements.append((
                            ' '.join(phrase_words),
                            corrected_phrase,
                            i,
                            i + phrase_len
                        ))
                except Exception:
                    continue
        
        # Apply replacements (process from end to start to preserve indices)
        if phrase_replacements:
            # Sort by start index (descending) to replace from end
            phrase_replacements.sort(key=lambda x: x[2], reverse=True)
            
            # Only apply the most suspicious replacement per sentence to avoid over-correction
            worst_replacement = phrase_replacements[0]
            original_phrase, corrected_phrase, start_idx, end_idx = worst_replacement
            
            # Preserve capitalization of the first word if needed
            if words[start_idx][0].isupper():
                corrected_words = corrected_phrase.split()
                if corrected_words:
                    corrected_words[0] = corrected_words[0].capitalize()
                corrected_phrase = ' '.join(corrected_words)
            
            # Replace in the sentence
            new_words = words[:start_idx] + corrected_phrase.split() + words[end_idx:]
            sentence = ' '.join(new_words)
        
        # Clean up any double spaces
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        corrected_sentences.append(sentence + punctuation)
    
    return ''.join(corrected_sentences).strip()


def _adapt_phrase_to_context(
    suspicious_words: list[str],
    reference_words: list[str],
    sentence_words: list[str],
    phrase_start_idx: int
) -> str:
    """
    Adapt a reference phrase to fit the context of the sentence.
    Tries to preserve sentence structure while using the correct semantic meaning.
    """
    # Extract the main action/verb from the reference
    # Common customer support actions: enter, click, check, verify, access, etc.
    action_verbs = ['enter', 'click', 'check', 'verify', 'access', 'update', 'change', 
                    'reset', 'track', 'cancel', 'contact', 'follow', 'navigate', 
                    'select', 'complete', 'submit', 'request', 'speak']
    
    # Find action verb in reference
    reference_action = None
    for word in reference_words:
        if word.lower() in action_verbs:
            reference_action = word.lower()
            break
    
    # If we found an action, try to construct a sensible phrase
    if reference_action:
        # Look at words before the suspicious phrase for context
        context_before = sentence_words[max(0, phrase_start_idx - 2):phrase_start_idx]
        
        # Build a corrected phrase using the action and context
        if context_before:
            # Try to preserve some context
            corrected = ' '.join(context_before[-1:]) + ' ' + reference_action
            # Add common object if available
            if 'password' in ' '.join(suspicious_words).lower():
                corrected += ' your password'
            elif 'button' in ' '.join(suspicious_words).lower() or 'click' in reference_action:
                corrected += ' the button'
            elif 'email' in ' '.join(suspicious_words).lower():
                corrected += ' your email'
            else:
                # Use a generic object from the reference
                if len(reference_words) > 1:
                    corrected += ' ' + ' '.join(reference_words[1:])
            return corrected.strip()
        else:
            # No context, just use the reference phrase
            return ' '.join(reference_words)
    else:
        # No clear action found, use the reference as-is
        return ' '.join(reference_words)


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
    Handles various edge cases including incomplete sentences, URLs, repetition, etc.
    """
    if not text or not text.strip():
        return text.strip()
    
    import re
    
    # Use spell checker for automatic typo correction
    try:
        from spellchecker import SpellChecker
        spell = SpellChecker()
        # Add domain-specific words to dictionary to avoid false positives
        domain_words = [
            'password', 'passwords', 'account', 'accounts', 'settings', 'privacy', 'security',
            'verification', 'authentication', 'subscription', 'cancellation', 'refund', 'refunds',
            'shipping', 'delivery', 'tracking', 'payment', 'billing', 'information', 'question',
            'questions', 'assistant', 'support', 'troubleshooting', 'feature', 'features',
            'notifications', 'preferences', 'order', 'orders', 'email', 'phone', 'address',
            'manager', 'supervisor', 'specialist', 'team', 'technical', 'billing', 'account',
            'reset', 'update', 'change', 'cancel', 'return', 'track', 'check', 'verify',
            'connect', 'transfer', 'escalate', 'help', 'assist', 'resolve', 'issue', 'problem'
        ]
        for word in domain_words:
            spell.word_frequency.load_words([word])
    except ImportError:
        # Fallback: if spellchecker not available, use minimal correction
        spell = None
    
    # Fix typos using spell checker (only for words that are clearly misspelled)
    if spell:
        words = text.split()
        fixed_words = []
        for word in words:
            # Extract word without punctuation
            word_match = re.match(r'([\w\']+)([^\w\']*)', word)
            if word_match:
                word_part = word_match.group(1)
                punctuation = word_match.group(2) if word_match.group(2) else ''
                
                # Skip very short words, numbers, and URLs
                if len(word_part) <= 2 or word_part.isdigit() or 'http' in word_part.lower():
                    fixed_words.append(word)
                    continue
                
                # Check if word is misspelled
                word_lower = word_part.lower()
                if word_lower not in spell:
                    # Get correction candidates
                    candidates = spell.candidates(word_lower)
                    if candidates:
                        # Use the most likely correction
                        correction = spell.correction(word_lower)
                        if correction and correction != word_lower:
                            # Preserve original capitalization
                            if word_part[0].isupper():
                                correction = correction.capitalize()
                            fixed_words.append(correction + punctuation)
                        else:
                            fixed_words.append(word)
                    else:
                        fixed_words.append(word)
                else:
                    fixed_words.append(word)
            else:
                fixed_words.append(word)
        
        text = ' '.join(fixed_words)
    
    # Normalize whitespace (multiple spaces, tabs, newlines)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove common markdown artifacts that might appear
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic
    text = re.sub(r'#+\s*', '', text)  # Headers
    text = re.sub(r'```[^`]*```', '', text)  # Code blocks
    text = re.sub(r'`([^`]+)`', r'\1', text)  # Inline code
    
    # Remove trailing incomplete words (common generation artifacts)
    text = re.sub(r'\s+\w{1,2}$', '', text)  # Remove 1-2 letter words at end
    
    # Detect and handle incomplete sentences at the end
    incomplete_patterns = [
        r'\b(?:based|according|depending|relying|accordingly)\s*$',
        r'\b(?:and|or|but|so|then|also|however|therefore|moreover)\s*$',
        r'\b(?:if|when|where|while|because|since|although|unless)\s*$',
        r'\b(?:the|a|an|this|that|these|those)\s*$',
        r'\b(?:to|for|with|from|by|at|in|on|of)\s*$',
    ]
    for pattern in incomplete_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            # Find the last complete sentence before this incomplete fragment
            # Look for sentence endings before the last 20 characters
            truncate_point = max(
                text.rfind('.', 0, len(text) - 20),
                text.rfind('!', 0, len(text) - 20),
                text.rfind('?', 0, len(text) - 20)
            )
            if truncate_point > 50:
                text = text[:truncate_point + 1].strip()
                break
    
    # Split into sentences (handle multiple sentence endings, but preserve abbreviations)
    # Use a more sophisticated sentence splitter that handles common abbreviations
    sentence_endings = re.compile(r'([.!?]+)\s+')
    sentences = sentence_endings.split(text)
    # Reconstruct sentences properly
    reconstructed = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            reconstructed.append(sentences[i] + sentences[i + 1])
        else:
            reconstructed.append(sentences[i])
    if len(sentences) % 2 == 1:
        reconstructed.append(sentences[-1])
    
    sentences = [s.strip() for s in reconstructed if s.strip()]
    
    # Limit to reasonable number of sentences for customer support (6-7 max for longer responses)
    # But be smarter about it - keep complete thoughts
    max_sentences = 7
    if len(sentences) > max_sentences:
        # Try to keep complete paragraphs/thoughts rather than just cutting
        # Look for natural break points (questions, transitions)
        keep_sentences = []
        for i, sent in enumerate(sentences[:max_sentences + 2]):
            if i < max_sentences:
                keep_sentences.append(sent)
            elif sent.strip().endswith('?'):
                # Keep questions even if over limit
                keep_sentences.append(sent)
                break
        sentences = keep_sentences[:max_sentences + 1]  # Allow one extra for questions
        # Rejoin with proper spacing
        text = ' '.join(sentences).strip()
    
    # Check for repetitive content (improved detection)
    if len(text) > 200:
        words = text.lower().split()
        if len(words) > 50:
            # Check for repeated 8-word sequences (more sensitive)
            for window_size in [8, 10, 12]:
                found_repetition = False
                for i in range(len(words) - window_size * 2):
                    seq = ' '.join(words[i:i+window_size])
                    # Check if this sequence appears later (with some tolerance)
                    later_text = ' '.join(words[i+window_size:])
                    if seq in later_text:
                        # Found repetition - truncate at the first occurrence
                        word_index = text.lower().find(seq, i + window_size * 5)  # Allow some overlap
                        if word_index != -1:
                            # Find the sentence boundary before this point
                            truncate_point = max(
                                text.rfind('.', 0, word_index),
                                text.rfind('!', 0, word_index),
                                text.rfind('?', 0, word_index)
                            )
                            if truncate_point > 50:  # Only truncate if we have enough content
                                text = text[:truncate_point + 1].strip()
                                found_repetition = True
                                break
                if found_repetition:
                    break
    
    # Fix nonsensical phrases using pattern matching and semantic coherence
    # First, use pattern matching for known weird phrases - replace with sensible alternatives
    nonsensical_replacements = [
        # "both fingers" patterns - replace with appropriate actions
        (r'\benter\s+([^\.]+?)\s+twice\s+using\s+both\s+fingers\b', r'enter \1 twice'),
        (r'\busing\s+both\s+fingers\s+to\s+([^\.]+?)\b', r'to \1'),
        (r'\bwith\s+both\s+fingers\b', ''),
        (r'\bboth\s+fingers\b', ''),
        (r'\bclick\s+([^\.]+?)\s+with\s+both\s+fingers\b', r'click \1'),
    ]
    
    for pattern, replacement in nonsensical_replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Then use semantic coherence checking for other weird phrases
    text = _check_semantic_coherence(text)
    
    # Clean up any double spaces created by removals
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove common rambling patterns and off-topic content
    # Look for phrases that suggest the response is going off-topic or rambling
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
        "are there any other ways",
        "you deserve the best",
        "i can't help with that",
        "that's not something i can",
        "i'm not able to",
        "i don't have access to",
        "i'm sorry but i",
        "unfortunately i cannot",
    ]
    
    # Enhanced rambling detection - look for repetitive sentence structures
    text_lower = text.lower()
    words = text_lower.split()
    
    # Check for excessive repetition of sentence starters
    sentence_starters = ["i", "you", "we", "they", "this", "that", "it", "if", "when", "to"]
    starter_counts = {}
    for i, word in enumerate(words):
        if word in sentence_starters and (i == 0 or words[i-1] in ['.', '!', '?'] or i < 3):
            starter_counts[word] = starter_counts.get(word, 0) + 1
    
    # If any starter appears too many times, it might be rambling
    if starter_counts and max(starter_counts.values()) > 8:
        # Find where excessive repetition starts
        for indicator in rambling_indicators:
            idx = text_lower.find(indicator)
            if idx > 200:  # Only if it's later in the response
                # Truncate at the sentence before this indicator
                truncate_point = max(
                    text.rfind('.', 0, idx),
                    text.rfind('!', 0, idx),
                    text.rfind('?', 0, idx)
                )
                if truncate_point > 50:  # Keep at least some content
                    text = text[:truncate_point + 1].strip()
                    break
    
    # Additional check: if response is very long and has many "I" statements, it might be rambling
    if len(text) > 600 and text_lower.count(" i ") > 10:
        # Find a good stopping point (after a question or clear instruction)
        for punct in ['?', '!']:
            last_punct = text.rfind(punct, 0, len(text) - 100)
            if last_punct > 300:
                text = text[:last_punct + 1].strip()
                break
    
    # Final length check - customer support responses can be longer now, but still reasonable
    max_chars = 800  # Increased from 500 to allow more detailed responses
    if len(text) > max_chars:
        # First, check if we're cutting through a URL - preserve URLs
        url_pattern = r'https?://[^\s]+|www\.[^\s]+'
        urls = list(re.finditer(url_pattern, text))
        if urls:
            # Find the last complete URL before max_chars
            for url_match in reversed(urls):
                if url_match.end() <= max_chars + 20:  # Allow some buffer
                    # URL is complete, truncate after it
                    truncate_point = url_match.end()
                    # Find next sentence boundary after URL
                    next_sentence = text.find('.', truncate_point, truncate_point + 50)
                    if next_sentence != -1:
                        text = text[:next_sentence + 1].strip()
                        break
                    elif truncate_point < max_chars + 50:
                        # Keep URL and truncate at next sentence
                        truncate_point = max(
                            text.rfind('.', truncate_point, max_chars + 50),
                            text.rfind('!', truncate_point, max_chars + 50),
                            text.rfind('?', truncate_point, max_chars + 50)
                        )
                        if truncate_point > truncate_point - 20:
                            text = text[:truncate_point + 1].strip()
                            break
        
        # Check if we're in a numbered list - don't cut off mid-list
        if len(text) > max_chars:
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
                    truncate_point = max(
                        text.rfind('.', 0, max_chars),
                        text.rfind('!', 0, max_chars),
                        text.rfind('?', 0, max_chars)
                    )
                    if truncate_point > 50:
                        text = text[:truncate_point + 1].strip()
                    else:
                        # Last resort: just truncate
                        text = text[:max_chars].strip()
                        if not text.endswith(('.', '!', '?')):
                            text += '.'
            else:
                # No numbered list, truncate at sentence boundary
                truncate_point = max(
                    text.rfind('.', 0, max_chars),
                    text.rfind('!', 0, max_chars),
                    text.rfind('?', 0, max_chars)
                )
                if truncate_point > 50:
                    text = text[:truncate_point + 1].strip()
                else:
                    # Fallback: just truncate
                    text = text[:max_chars].strip()
                    if not text.endswith(('.', '!', '?')):
                        text += '.'
    
    # Final cleanup: ensure proper sentence ending
    text = text.strip()
    if text and not text.endswith(('.', '!', '?', ':', ';')):
        # Only add period if the last character is a letter or number
        if text and text[-1].isalnum():
            text += '.'
    
    # Remove any trailing incomplete fragments
    text = re.sub(r'\s+\w{1,2}\.$', '.', text)  # Remove 1-2 letter words before period
    
    return text.strip()


@app.on_event("startup")
def load_model() -> None:
    global model, tokenizer, device, supports_chat_template, system_prompt, default_max_new_tokens

    model_id = os.getenv("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    adapter_dir_env = os.getenv("ADAPTER_DIR", "outputs/smoke_001")
    device_env = os.getenv("DEVICE", "auto")
    default_max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "1000"))  # High default, can be overridden

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
    # Allow high token limits - only cap at reasonable maximum for safety
    max_new_tokens = payload.max_new_tokens or default_max_new_tokens
    max_new_tokens = min(max_new_tokens, 2000)  # Safety cap at 2000 tokens
    temperature = payload.temperature if payload.temperature is not None else 0.6
    top_p = payload.top_p if payload.top_p is not None else 0.85
    repetition_penalty = payload.repetition_penalty if payload.repetition_penalty is not None else 1.5

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
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
