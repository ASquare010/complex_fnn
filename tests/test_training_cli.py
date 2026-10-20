"""Synthetic lifecycle tests only; no corpus training or research updates."""
import contextlib
import io
import json
from pathlib import Path
import uuid
import unittest
from unittest.mock import patch
import torch
from models.training_cli import ROOT, execute, load_parent, parser_for, recipe_for, schedule_lr
from models.training_tasks import Task

class TinyTask:
    """Small differentiable test stand-in with variable-sized target groups."""

    def __init__(self, recipe, device):
        self.model = torch.nn.Linear(2, 2).to(device)
        self.codec = None
        self.device = device

    def examples(self, rows):
        return [r['tokens'].float().reshape(-1, 2) for r in rows]

    def count(self, items):
        return sum((len(x) for x in items))

    def loss(self, items):
        x = torch.cat(items).to(self.device)
        y = x[:, 0].long() % 2
        logits = self.model(x)
        return (torch.nn.functional.cross_entropy(logits, y), {'tokens': len(x), 'correct': 0, 'exact': 0, 'sequences': len(items)})

    @torch.no_grad()
    def evaluate(self, examples, microbatch):
        loss, stats = self.loss(examples)
        return {'ce': float(loss), 'token_accuracy': 0.0, **stats}

class TrainingCliTests(unittest.TestCase):

    def args(self, *extra):
        return parser_for().parse_args(['branch', '--device', 'cpu', '--steps', '4', '--microbatch', '1', '--accumulation', '2', '--eval-every', '4', '--save-every', '2', *extra])

    def test_invalid_and_schedule(self):
        for opts in [('--steps', '0'), ('--microbatch', '0'), ('--width', '7'), ('--lr', 'nan'), ('--warmup', '-1')]:
            with self.assertRaises(ValueError):
                recipe_for(self.args(*opts))
        r = recipe_for(self.args())
        parent = {'recipe': r, 'step': 2}
        resumed = recipe_for(self.args(), parent)
        self.assertEqual(schedule_lr(r['schedule'], 3), schedule_lr(resumed['schedule'], 3))
        extension = recipe_for(self.args('--steps', '8'), parent)
        self.assertEqual(extension['schedule']['start'], 2)
        with self.assertRaises(ValueError):
            recipe_for(self.args('--width', '128'), parent)

    def test_resume_matches_uninterrupted_and_rejects_corruption(self):
        folder = ROOT / 'dump' / ('cli-test-' + uuid.uuid4().hex)
        folder.mkdir()
        with contextlib.nullcontext(folder):
            base = Path(folder)
            data = base / 'data'
            data.mkdir()
            from shutil import copyfile
            copyfile(ROOT / 'dump/pretrained-latent64-fced6bd/tokenizer.json', data / 'tokenizer.json')
            rows = {'train': [{'tokens': torch.tensor([1, 0, 0, 1])}, {'tokens': torch.tensor([0, 1])}], 'valid': [{'tokens': torch.tensor([1, 0])}]}
            args = self.args('--data', str(data))
            with patch('models.training_tasks.Task', TinyTask), patch('models.training_data.read_source', return_value=(rows, {'synthetic': True})), contextlib.redirect_stdout(io.StringIO()):
                full = base / 'full'
                execute(args, recipe_for(args), full)
                paused = base / 'paused'
                args.stop_after = 2
                execute(args, recipe_for(args), paused)
                path, parent = load_parent(paused)
                self.assertEqual(parent['step'], 2)
                self.assertEqual(json.loads((paused / 'status.json').read_text())['state'], 'paused')
                args.stop_after = None
                resumed = base / 'resumed'
                execute(args, recipe_for(args, parent), resumed, path, parent)
                _, a = load_parent(full)
                _, b = load_parent(resumed)
                self.assertEqual(b['run_name'], 'resumed')
                self.assertEqual(b['recipe']['run_name'], 'resumed')
                self.assertEqual(b['recipe']['parent_run_name'], 'paused')
                for key in a['model']:
                    self.assertTrue(torch.equal(a['model'][key], b['model'][key]))
                self.assertEqual(a['python_rng'], b['python_rng'])
                self.assertTrue(torch.equal(a['torch_rng'], b['torch_rng']))
                (resumed / 'last.sha256').write_text('wrong')
                with self.assertRaises(ValueError):
                    load_parent(resumed)

    def test_real_branch_and_compressor_objectives(self):
        for kind in ('branch', 'compressor'):
            args = parser_for().parse_args([kind, '--device', 'cpu', '--width', '16', '--heads', '2', '--layers', '1', '--context', '64'])
            task = Task(recipe_for(args), 'cpu')
            items = task.examples([{'tokens': torch.randint(3, 4096, (2, 64))}])
            loss, stats = task.loss(items)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(stats['tokens'], 0)
            self.assertTrue(any((p.grad is not None for p in task.model.parameters())))

    def test_accumulation_is_target_weighted(self):
        torch.manual_seed(0)
        a = TinyTask({}, 'cpu')
        b = TinyTask({}, 'cpu')
        b.model.load_state_dict(a.model.state_dict())
        items = [torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([[0.0, 1.0]])]
        a.loss(items)[0].backward()
        for item in items:
            loss, stats = b.loss([item])
            (loss * stats['tokens'] / b.count(items)).backward()
        for p, q in zip(a.model.parameters(), b.model.parameters()):
            torch.testing.assert_close(p.grad, q.grad)
if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
