from __future__ import annotations

try:
    # `trainers.*` package로 import되는 local runner 경로
    from .nnUNetTrainerOLES3DB0Development import (
        nnUNetTrainerOLES3DB0Development,
    )
except ImportError:
    # nnU-Net external-trainer discovery가 파일을 top-level module로 로드하는 경로
    from nnUNetTrainerOLES3DB0Development import (
        nnUNetTrainerOLES3DB0Development,
    )


class nnUNetTrainerOLES3DB0Main(nnUNetTrainerOLES3DB0Development):
    """Cloud main comparison용 B0 30k trainer 식별자."""
