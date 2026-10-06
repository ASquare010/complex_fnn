"""The direct decoder cannot access source tokens or a teacher-forced prefix."""

import tempfile
import unittest
from pathlib import Path

import torch

from models.position_compressor.codec import Codec
from models.position_compressor.data import tokenizer_for
from models.position_compressor.transformer import Config, Model


class PositionCompressorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.model = Model(
            Config(width=32, heads=4, encoder_layers=2, decoder_layers=1, max_tokens=32)
        ).eval()

    def test_padding_and_latent_only_reconstruction(self):
        x = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10, 11]])
        z, lengths = self.model.encode(x, x.ne(0))
        expected = self.model.decode(z, lengths)
        padded = torch.cat([x, torch.full((1, 7), 90)], 1)
        other, sizes = self.model.encode(padded, torch.arange(16)[None] < 9)
        torch.testing.assert_close(z, other, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(expected, self.model.decode(other, sizes), atol=1e-6, rtol=1e-5)
        del x, padded
        self.assertEqual(self.model.generate(z, lengths).shape, (1, 9))

    def test_gradient_and_shape_contract(self):
        x = torch.tensor([[3, 4, 5], [6, 7, 0]])
        result = self.model(x, x.ne(0))
        result.square().mean().backward()
        for name, p in self.model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        with self.assertRaises(ValueError):
            self.model.encode(
                torch.ones(1, 33, dtype=torch.long), torch.ones(1, 33, dtype=torch.bool)
            )
        with self.assertRaises(ValueError):
            self.model.decode(torch.zeros(1, 1, 32), torch.tensor([9]))

    def test_encoder_only_chunking_and_saved_memories(self):
        tokenizer = tokenizer_for(["names numbers and Unicode"], 259)
        codec = Codec(self.model, tokenizer)
        text = " Amina: 17!\n" + "é🙂\t<eos>" * 20
        payload = codec.encode(text)
        self.assertGreater(len(payload["chunks"]), 1)
        self.assertEqual(set(payload), {"format", "decoder_id", "tokenizer_id", "chunks"})
        for chunk in payload["chunks"]:
            self.assertEqual(set(chunk), {"vectors", "lengths"})
            self.assertLessEqual(int(chunk["lengths"][0]), 32)
        expected = codec.decode(payload)
        Path("dump").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir="dump") as tmp:
            encoder_path = codec.export_encoder(Path(tmp) / "encoder.pt")
            encoder = Codec.load_encoder(encoder_path)
            self.assertLess(
                sum(p.numel() for p in encoder.model.parameters()),
                sum(p.numel() for p in codec.model.parameters()),
            )
            self.assertFalse(
                any(k.startswith(("decoder", "expand")) for k in encoder.model.state_dict())
            )
            other = encoder.encode(text)
            for a, b in zip(payload["chunks"], other["chunks"]):
                torch.testing.assert_close(a["vectors"], b["vectors"], rtol=0, atol=0)
                torch.testing.assert_close(a["lengths"], b["lengths"], rtol=0, atol=0)
            self.assertEqual(codec.decode(other), expected)
            with self.assertRaises(ValueError):
                encoder.decode(other)
            path = encoder.save_memory(text, Path(tmp) / "memories.pt")
            self.assertEqual(codec.recover_file(path), expected)
        self.assertEqual(codec.reconstruct("")["output"], "")
        payload["decoder_id"] = "wrong"
        with self.assertRaises(ValueError):
            codec.decode(payload)


if __name__ == "__main__":
    unittest.main()
