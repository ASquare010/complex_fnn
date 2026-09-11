"""Preload in the successful training order, then execute the unchanged audit."""

import sympy
import torch
import torch._dynamo


def main():
    print(f"CPU preload passed: sympy={sympy.__version__}, torch={torch.__version__}", flush=True)
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from results.whole_job_memory_workspace_v1.source.audit import run

    run()


if __name__ == "__main__":
    main()
