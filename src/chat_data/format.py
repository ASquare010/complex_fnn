"""Shared token-exact chat serializer for preparation and future inference."""
ROLES = {"system": "System", "user": "User", "assistant": "Assistant"}


def encode_dialogue(messages, tokenizer):
    """Same textual roles as inference; EOS2 after every assistant turn.

    Tokenization is explicitly per header/body. Inference must use this serializer,
    including identical newline handling, rather than re-tokenize a joined string.
    """
    tokens, mask, spans = [], [], []
    for i, message in enumerate(messages):
        role = message['role']
        header = ('\n' if i else '') + ROLES[role] + ':'
        head = tokenizer.encode(header).ids
        body = tokenizer.encode(' ' + message['content']).ids
        start = len(tokens)
        tokens.extend(head)
        mask.extend([0] * len(head))
        body_start = len(tokens)
        tokens.extend(body)
        mask.extend([int(role == 'assistant')] * len(body))
        body_end = len(tokens)
        if role == 'assistant':
            tokens.append(2)
            mask.append(1)
        spans.append(dict(message=i, role=role, start=start, body_start=body_start,
                          body_end=body_end, end=len(tokens)))
    return tokens, mask, spans


def encode_prompt(messages, tokenizer):
    """Complete prior turns and final user message, then the assistant header."""
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("Prompt must end with a user message")
    tokens, _, _ = encode_dialogue(messages, tokenizer)
    return tokens + tokenizer.encode("\nAssistant:").ids
