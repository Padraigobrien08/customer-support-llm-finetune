# Dataset Format Specification

## Format: JSONL (JSON Lines)

Each line in the dataset file is a valid JSON object representing a single training example.

## Schema

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful customer support assistant."},
    {"role": "user", "content": "I need help with my order."},
    {"role": "assistant", "content": "I'd be happy to help you with your order. Can you provide your order number?"},
    {"role": "user", "content": "It's #12345"},
    {"role": "assistant", "content": "Thank you. I can see your order is currently being processed. It should ship within 2-3 business days."}
  ]
}
```

### Message Object Structure

- **`role`** (string, required): One of `"system"`, `"user"`, or `"assistant"`
- **`content`** (string, required): The message text content

### Rules

- Each example must contain at least one `user` message and one `assistant` message
- `system` messages are optional and should appear only at the start of the conversation
- Messages must be ordered chronologically
- Multi-turn conversations are supported by including multiple alternating `user`/`assistant` pairs

## Example File (example.jsonl)

```
{"messages": [{"role": "system", "content": "You are a helpful customer support assistant."}, {"role": "user", "content": "When will my order arrive?"}, {"role": "assistant", "content": "I can check the status of your order. Could you please provide your order number?"}]}
{"messages": [{"role": "user", "content": "I want to return an item"}, {"role": "assistant", "content": "I can help you with that return. What's the reason for the return?"}, {"role": "user", "content": "It doesn't fit"}, {"role": "assistant", "content": "No problem. I'll process your return. You'll receive a return label via email within 24 hours."}]}
```

## Rationale

This schema was chosen for the following reasons:

1. **API Compatibility**: The `messages` array format with `role` and `content` fields matches the structure used by major chat-based LLM APIs (OpenAI, Anthropic, etc.), ensuring seamless integration with existing tooling and frameworks.

2. **Multi-turn Support**: The sequential message array naturally represents conversational context, allowing the model to learn from extended dialogues rather than isolated question-answer pairs.

3. **Framework Agnostic**: JSONL is widely supported across fine-tuning frameworks and can be easily converted to other formats if needed.

4. **Streaming-Friendly**: JSONL format allows processing large datasets line-by-line without loading entire files into memory.

5. **Extensibility**: Additional metadata fields can be added to each example object without breaking compatibility with existing parsers.

