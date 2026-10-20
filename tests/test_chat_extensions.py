"""Extension adapters must preserve known rows and associate observations safely."""
import copy
import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from extend_unified_chat import adapt_extended, schema_list
from prepare_unified_chat import adapt


def example():
    schema = dict(type='function', function=dict(name='lookup', parameters=dict(required=['name'])))
    return dict(messages=[dict(role='user', content='Find the value for this name.'),
        dict(role='assistant', content='<tool_call>{"name":"lookup","arguments":{"name":"Ana"}}</tool_call>'),
        dict(role='tool', content='<tool_response>{"name":"lookup","content":"red"}</tool_response>'),
        dict(role='assistant', content='The value is red.')],
        chat_template_kwargs=dict(enable_thinking=False, xml_tools=['<tools>'+json.dumps([schema])+'</tools>']))


def test_existing_row_identity_and_wrapped_schema():
    row = example()
    before = copy.deepcopy(row)
    assert adapt_extended(row) == adapt(row)
    row['chat_template_kwargs']['xml_tools'][0] = 'Use <tools></tools> schemas.\n'+row['chat_template_kwargs']['xml_tools'][0]+'\nReturn valid calls.'
    messages, thinking = adapt_extended(row)
    assert not thinking and 'Return valid calls.' in messages[0]['content']
    assert 'Use <tools></tools>' in messages[0]['content']
    assert before == example()


def test_plain_observation_requires_unambiguous_call():
    row = example()
    row['messages'][2]['content'] = 'Observation: red'
    messages, _ = adapt_extended(row)
    assert 'Observation: red' in messages[3]['content']
    row['messages'][1]['content'] *= 2
    with pytest.raises(ValueError, match='ambiguous_tool_observation'):
        adapt_extended(row)


def test_json_lines_and_reasoning_preservation():
    row = example()
    schema = json.loads(row['chat_template_kwargs']['xml_tools'][0][7:-8])[0]
    assert schema_list(json.dumps(schema)+'\n'+json.dumps(schema)) == [schema,schema]
    row['chat_template_kwargs']['xml_tools'] = [json.dumps(schema)]
    row['chat_template_kwargs']['enable_thinking'] = True
    row['messages'][1]['content'] = '<think>Look up the stored value.</think>'+row['messages'][1]['content']
    messages, thinking = adapt_extended(row)
    assert thinking and '<think>Look up the stored value.</think>' in messages[2]['content']
    row['messages'][1]['content'] = '<think>unfinished'
    with pytest.raises(ValueError, match='incomplete_reasoning'):
        adapt_extended(row)
