"""Regression checks against pre-cleanup outputs from the retained checkpoints."""
import unittest
from pathlib import Path
import torch
from storage import digest
from models.branch_sigmoid.loader import load_winner, WINNER
from models.position_compressor.codec import Codec


class SelectedModelsTests(unittest.TestCase):
    @unittest.skipUnless(Path("dump/selected-cleanup-20261007/parity.pt").exists(), "Local historical weights and parity receipt required")
    def test_original_checkpoint_outputs_and_counts(self):
        torch.set_num_threads(4)
        saved = torch.load('dump/selected-cleanup-20261007/parity.pt', weights_only=True)
        self.assertEqual(digest(WINNER), saved['branch_hash'])
        self.assertEqual(digest('dump/compression-span32-v1/encoder.pt'), saved['encoder_hash'])
        self.assertEqual(digest('dump/compression-span32-v1/last.pt'), saved['full_hash'])
        lm, _ = load_winner()
        self.assertEqual(sum(p.numel() for p in lm.parameters()), 8654208)
        self.assertEqual(sum(p.numel() for b in lm.blocks for p in b.ffn.parameters()),4718592)
        with torch.no_grad():
            torch.testing.assert_close(lm(saved['tokens']), saved['logits'], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'Legacy CurveFFN'):
            Codec.load_encoder('dump/compression-span32-v1/encoder.pt')
        with self.assertRaisesRegex(ValueError, 'Legacy CurveFFN'):
            Codec.load('dump/compression-span32-v1/last.pt')

    def test_selected_trainer_imports(self):
        from models.branch_sigmoid.training import run, evaluate
        from models.position_compressor.training import Session
        self.assertTrue(callable(run) and callable(evaluate) and Session is not None)

    @unittest.skipUnless(Path("dump/selected-cleanup-20261007/before.zip").exists(), "Local historical source archive required")
    def test_token_batches_preserve_targets_without_memory_input(self):
        import ast
        import zipfile
        from models.branch_sigmoid.data import batch
        with zipfile.ZipFile('dump/selected-cleanup-20261007/before.zip') as archive:
            text = archive.read('src/models/context_comparison/data.py').decode()
        function = next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='batch')
        namespace = {'torch':torch}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'historical-batch','exec'),namespace)
        row={'ids':torch.arange(600,dtype=torch.int32),'score':torch.arange(600)%3!=0}
        expected=namespace['batch']([(row,0),(row,1)],False,'cpu')
        actual=batch([(row,0),(row,1)],False,'cpu')
        self.assertIsNone(actual[2])
        for i in (0,1,3):torch.testing.assert_close(actual[i],expected[i],rtol=0,atol=0)


if __name__ == '__main__': unittest.main()
