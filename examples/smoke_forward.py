from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
    import torch.nn  # noqa: F401
except Exception as exc:
    raise SystemExit(
        "PyTorch with torch.nn is required. Install dependencies with "
        "`pip install -r requirements.txt` before running this smoke test."
    ) from exc

from msgm import MSGM, count_parameters


def main() -> None:
    model = MSGM(
        num_channels=62,
        num_features=7,
        num_classes=2,
        scale_lengths=(16, 23, 36, 76),
    )
    inputs = [torch.randn(2, length, 62, 7) for length in model.scale_lengths]
    logits = model(*inputs)
    print("62-channel logits:", tuple(logits.shape))
    print("trainable parameters:", count_parameters(model))

    model_32 = MSGM(
        num_channels=32,
        num_features=7,
        num_classes=2,
        scale_lengths=(16, 36, 76),
    )
    inputs_32 = [torch.randn(2, length, 32, 7) for length in model_32.scale_lengths]
    logits_32 = model_32(inputs_32)
    print("32-channel logits:", tuple(logits_32.shape))


if __name__ == "__main__":
    main()
