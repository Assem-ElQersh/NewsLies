import numpy as np
from torch.optim.lr_scheduler import LambdaLR


def get_scheduler(optimizer, warmup_steps: int = 0, total_steps: int = None,
                  schedule_type: str = "cosine", min_lr_ratio: float = 0.0):
    """Linear warmup followed by cosine decay to min_lr_ratio * base lr."""
    if total_steps is None:
        raise ValueError("total_steps is required for the decay schedule.")

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1 + float(np.cos(np.pi * progress)))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)
