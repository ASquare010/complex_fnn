import pytest
import sympy  # noqa: F401
import torch
import torch._dynamo  # noqa: F401

torch.set_num_threads(4)
raise SystemExit(pytest.main(["-q", "tests/test_training_memory.py"]))
