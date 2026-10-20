"""Check mode controls, observed tool masking and complete tool-call adaptation."""
import sys
import json
from pathlib import Path
import pytest
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from prepare_unified_chat import adapt, clean
from chat_data.unified_format import encode_dialogue, encode_prompt, with_mode


def test_tool_round_trip_and_observation_mask():
    row = dict(messages=[dict(role='user', content='Add two and three.'),
        dict(role='assistant', content="<tool_call>{'name':'add','arguments':{'a':2,'b':3}}</tool_call>"),
        dict(role='tool', content='<tool_response>{"name":"add","content":5}</tool_response>'),
        dict(role='assistant', content='The answer is five.')],
        chat_template_kwargs=dict(enable_thinking=False, xml_tools=[
        '<tools>[{"type":"function","function":{"name":"add","parameters":{"required":["a","b"]}}}]</tools>']))
    messages, thinking = adapt(row)
    assert not thinking and clean(messages) == messages
    assert '"name": "add"' in messages[2]['content']
    tokenizer = Tokenizer.from_file(str(ROOT/'dump/data/token-memory/tokenizer.json'))
    ids, mask, spans = encode_dialogue(messages, tokenizer)
    for span in spans:
        scored = mask[span['body_start']:span['end']]
        assert all(v == int(span['role']=='assistant') for v in scored)
    # Prefix used for actual inference is exactly the training prefix.
    prompt = row['messages'][:-1]
    # Canonicalized previous turns, without the automatically added system mode.
    prefix = messages[:-1]
    tokens, _, _ = encode_dialogue(prefix, tokenizer)
    assert ids[:spans[-1]['body_start']] == tokens + tokenizer.encode('\nAssistant:').ids
    row['messages'][1]['content'] = '<tool_call>{"name":"not_available","arguments":{}}</tool_call>'
    with pytest.raises(ValueError, match='invalid_tool_call'):
        adapt(row)


def test_reasoning_controls_do_not_fabricate_traces():
    messages = [dict(role='user', content='What is two plus three?'),
                dict(role='assistant', content='<think>Adding two to three gives five.</think>Five.')]
    adapted, mode = adapt(dict(messages=messages, chat_template_kwargs=dict(enable_thinking=True)))
    assert mode and adapted[0]['content']=='Reasoning: on.'
    with pytest.raises(ValueError, match='reasoning_mode'):
        adapt(dict(messages=messages, chat_template_kwargs=dict(enable_thinking=False)))
    tokenizer = Tokenizer.from_file(str(ROOT/'dump/data/token-memory/tokenizer.json'))
    prompt = messages[:1]
    off = encode_prompt(prompt, tokenizer, False)
    on = encode_prompt(prompt, tokenizer, True)
    assert off != on
    encoded, _, spans = encode_dialogue(with_mode(messages, True), tokenizer)
    assert encoded[:spans[-1]['body_start']] == on


def test_exact_source_migration_does_not_allow_model_changes():
    from chat_data.training import check_sources, sources
    migration = json.loads((ROOT/'src/chat_data/source_migration.json').read_text())
    current = sources()
    check_sources(current, current)
    # Retired raw512 sources cannot silently resume the new native256 model.
    with pytest.raises(ValueError, match='Sources changed'):
        check_sources(migration['before'], current)
    changed = {k.replace(chr(92),'/'):v for k,v in sources().items()}
    changed['src/models/token_memory/transformer.py'] = 'different-model'
    with pytest.raises(ValueError, match='Sources changed'):
        check_sources(current, changed)
