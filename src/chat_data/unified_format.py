"""Unified explicit-mode chat format, including observed tool responses.

Ordinary tokenizer text, no added vocabulary or model parameters. A mode is an
instruction learned from examples, not a guarantee of reasoning correctness.
"""
ROLES = {'system': 'System', 'user': 'User', 'assistant': 'Assistant', 'tool': 'Tool'}


def with_mode(messages, thinking=False):
    messages = [dict(m) for m in messages]
    instruction = 'Reasoning: ' + ('on' if thinking else 'off') + '.'
    if messages and messages[0]['role'] == 'system':
        messages[0]['content'] = instruction + '\n' + messages[0]['content']
    else:
        messages.insert(0, dict(role='system', content=instruction))
    return messages


def encode_dialogue(messages, tokenizer):
    tokens, mask, spans = [], [], []
    for i, message in enumerate(messages):
        role = message['role']
        start = len(tokens)
        head = tokenizer.encode(('\n' if i else '') + ROLES[role] + ':').ids
        body = tokenizer.encode(' ' + message['content']).ids
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


def encode_prompt(messages, tokenizer, thinking=False):
    if not messages or messages[-1]['role'] not in ('user', 'tool'):
        raise ValueError('Prompt must end with a user message or observed tool result')
    tokens, _, _ = encode_dialogue(with_mode(messages, thinking), tokenizer)
    return tokens + tokenizer.encode('\nAssistant:').ids
