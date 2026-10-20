"""CLI checkpoint/resume and readable periodic validation, without real training data."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from models.token_memory import training as t
import test_token_memory_canonical as fixture


class FakeData:
    def __init__(self, *args):
        self.per_epoch, self.total = 2, 4
        self.rows = np.tile(np.array([0, 0, 4, 4, 0]), (64, 1))
        self.datasets = {'train': self}

    def order(self, epoch):
        result = list(range(64))
        random.Random(47+epoch).shuffle(result)
        return result

    def batch(self, split, indices, arm, device):
        g = torch.Generator().manual_seed(17)
        ids = torch.randint(3, 4096, (len(indices), 1, 64), generator=g)
        from models.token_memory.wordbits import pack_values
        memory = pack_values(torch.ones(len(indices), 1, 256), ids)
        x = torch.tensor([[1, 3, 4, 5]]).expand(len(indices), -1).clone()
        y = torch.tensor([[3, 4, 5, 2]]).expand_as(x).clone()
        return (memory, torch.ones(len(indices), 1, dtype=torch.bool), x, y,
                torch.ones_like(x, dtype=torch.bool), torch.ones(len(indices), 1, dtype=torch.long))


class FakeLabels:
    def counts(self, indices):
        return {kind: dict(segments=0, terms=0) for kind in ('query', 'source')}

    def batch(self, indices, device):
        return {kind: torch.empty(0, 4, dtype=torch.long) for kind in ('query', 'source')}


class Checks(unittest.TestCase):
    def test_actual_cli_resume_matches_uninterrupted_and_periodic_eval(self):
        fixture.Checks.setUpClass()
        f = fixture.Checks()
        evaluations = []
        def evaluate(model, data, folder, step, micro, device):
            evaluations.append((folder.name, step))
            print(t.validation_text(step, dict(ce=2., token_accuracy=.5, correct=2, targets=4)))
        def args(*extra):
            return t.parser().parse_args(['--device', 'cpu', '--microbatch', '4', '--eval-every', '1', *extra])
        workspace_temp = (t.ROOT/'dump').resolve()
        with tempfile.TemporaryDirectory(dir=workspace_temp, prefix='canonical-test-') as temp, redirect_stdout(io.StringIO()) as output:
            root = Path(temp)
            with (patch.object(t, 'ROOT', root), patch.object(t, 'Data', FakeData),
                  patch.object(t, 'Labels', FakeLabels), patch.object(t, 'assets', return_value={}),
                  patch.object(t, 'source_hashes', return_value={}), patch.object(t, 'Model', side_effect=lambda: f.model()),
                  patch.object(t, 'evaluate', side_effect=evaluate)):
                t.execute(args('--name', 'split', '--stop-after', '1'))
                first = t.load(root/'dump/split')
                self.assertEqual(first['step'], 1)
                t.execute(args('--resume', str(root/'dump/split'), '--stop-after', '1'))
                t.execute(args('--name', 'whole', '--stop-after', '2'))
                split, whole = (t.load(root/'dump'/n) for n in ('split', 'whole'))
                self.assertEqual(split['cursor'], dict(epoch=1, batch_index=0))
                for key, value in split['model'].items():
                    torch.testing.assert_close(value, whole['model'][key], atol=0, rtol=0)
                for key, value in split['optimizer']['state'].items():
                    for field, tensor in value.items():
                        torch.testing.assert_close(tensor, whole['optimizer']['state'][key][field], atol=0, rtol=0)
            self.assertEqual(evaluations, [('split', 1), ('split', 2), ('whole', 1), ('whole', 2)])
            text = output.getvalue()
            self.assertIn('accuracy 50.000%', text)
            self.assertIn('ETA', text)
            self.assertNotIn("{'", text)

    def test_cli_counts_fixed_recipe_and_human_validation(self):
        self.assertEqual(t.recipe('a', 'b')['epochs'], 2)
        self.assertEqual(t.recipe('a', 'b')['width'], 256)
        with self.assertRaises(ValueError):
            t.checked_counts(t.parser().parse_args(['--microbatch', '3']))
        self.assertNotIn('{', t.validation_text(1000, dict(ce=2.5, token_accuracy=.6, correct=60, targets=100)))

    def test_generation_real_cache_empty_history_and_64_token_boundaries(self):
        from types import SimpleNamespace
        from models.token_memory.generation import generate
        fixture.Checks.setUpClass()
        for length, advances in ((0, 3), (63, 3), (65, 65)):
            model = fixture.Checks().model().eval()
            calls = []
            def encoder(tokens, valid):
                calls.append(tokens.clone())
                return tokens.float().mean(-1)[:, None, None].expand(-1, 1, 256), None
            def fixed_output(h, source, probability, null, embedding):
                values = h.new_full((*h.shape[:2], 4096), -100.)
                values[..., 5] = 0
                return values
            codec = SimpleNamespace(device='cpu', model=encoder)
            history = torch.arange(length) % 30 + 3
            with patch.object(model.reader, 'distribution', side_effect=fixed_output):
                actual = generate(model, codec, history, advances)
            self.assertEqual(actual, [5]*advances)
            self.assertTrue(all(v.shape[1] == 64 for v in calls))
            if length == 63:
                torch.testing.assert_close(calls[0][0], torch.cat((history, torch.tensor([5]))))
            elif length == 65:
                torch.testing.assert_close(calls[0][0], history[:64])
                torch.testing.assert_close(calls[1][0], torch.cat((history[64:], torch.full((63,), 5))))
            else:
                self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
