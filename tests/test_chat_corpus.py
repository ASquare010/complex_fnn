import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from prepare_chat_corpus import clean_messages, encode_dialogue, identity, memory_episode, split_for, Writer
from chat_data.dataset import Dataset, batch, batches_for_epoch, update_count
from chat_data.format import encode_prompt


def tokenizer():
    return Tokenizer.from_file(str(ROOT / 'dump/data/token-memory/tokenizer.json'))


def test_whole_turns_and_assistant_only_targets():
    t = tokenizer()
    messages = [dict(role='user', content='Hello!'), dict(role='assistant', content='Hello.'),
                dict(role='user', content='My name is Amina.'), dict(role='assistant', content='Nice to meet you, Amina.')]
    tokens, mask, spans = encode_dialogue(clean_messages(messages), t)
    assert encode_prompt(messages[:-1], t) == tokens[:spans[-1]['body_start']]
    for span in spans:
        assert not any(mask[span['start']:span['body_start']])
        if span['role'] == 'assistant':
            assert all(mask[span['body_start']:span['end']])
            assert tokens[span['end'] - 1] == 2
        else:
            assert not any(mask[span['start']:span['end']])
    with pytest.raises(ValueError):
        clean_messages(messages[:-1])


def test_memory_labels_and_old_fact_boundary():
    t = tokenizer()
    for split in ('train', 'dev', 'test'):
        for i in (10, 11):
            messages, events = memory_episode(i, split)
            tokens, mask, spans = encode_dialogue(messages, t)
            assert len(tokens) <= 2560
            assert spans[0]['body_end'] <= (spans[5]['body_start'] // 512) * 512
            assert events['read']['no_match'] == (i % 5 == 0)
            assert len(events['read']['positive_record_ids']) == (0 if i % 5 == 0 else 1)
            # The question does not reveal the randomly assigned notebook location.
            for record in events['records']:
                assert record['value'] not in messages[4]['content']


def test_epoch_coverage_without_replacement():
    for epoch in range(2):
        draws = list(batches_for_epoch(65, 32, epoch=epoch))
        assert sorted(sum(draws, [])) == list(range(65))
        assert [len(b) for b in draws] == [32, 32, 1]
    assert update_count(65) == 6
    assert list(batches_for_epoch(65)) != list(batches_for_epoch(65, epoch=1))


def test_writer_split_and_batch_do_not_mask_user_attention():
    tmp_path = ROOT / 'dump' / ('chat-data-test-' + uuid.uuid4().hex)
    tmp_path.mkdir()
    t = tokenizer()
    writer = Writer(tmp_path, t, 2560, set())
    messages = [dict(role='user', content='Where did I put the green notebook yesterday?'),
                dict(role='assistant', content="I don't know.")]
    group = identity(messages[0]['content'])
    assert writer.add(messages, 'test-source', {}, group=group, split='train')
    assert not writer.add(messages, 'test-source', {}, group=group, split='dev')
    writer.close()
    (tmp_path / 'manifest.json').write_text('{"format":"chat-memory-v2"}')
    ds = Dataset(tmp_path)
    codec = SimpleNamespace(device='cpu')
    mem, mm, x, y, valid = batch([ds[0]], codec)
    assert valid.all() and (y == -100).any() and (y != -100).any()
    assert mem.shape[1] == 0
    assert x[0, 0] == 1
    assert torch.equal(x[0, 1:], ds[0][1][:-1])
