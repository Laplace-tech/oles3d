#!/usr/bin/env python3
"""B0/B1/A1/P comparator fairness와 allocation CPU cost 감사."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = PROJECT_ROOT / "research"
NNUNET_RESEARCH_ROOT = RESEARCH_ROOT / "nnunet"
for path in (RESEARCH_ROOT, NNUNET_RESEARCH_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sampling.candidate_pools import (  # noqa: E402
    ERROR_TYPES,
    ErrorCandidatePools,
    choose_b1_static_candidate,
)
from sampling.error_type_learning_state import (  # noqa: E402
    ErrorTypeLearningState,
    choose_p_adaptive_candidate,
)
from sampling.organ_learning_state import (  # noqa: E402
    OrganLearningState,
    choose_a1_adaptive_candidate,
)


DATASET_NAME = "Dataset501_OLES3D9Organs"
NNUNET_ROOT = PROJECT_ROOT / "data" / "nnunet"
PREPROCESSED_ROOT = NNUNET_ROOT / "nnUNet_preprocessed" / DATASET_NAME
FROZEN_MANIFEST = (
    PROJECT_ROOT / "artifacts/data_foundation/1_7d_data_manifest.json"
)


def parse_arguments() -> argparse.Namespace:
    """Microbenchmark 반복 수와 출력 경로 해석."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws-per-method", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=55_254)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def synthetic_pools() -> ErrorCandidatePools:
    """9 organs×3 types가 모두 nonempty인 bounded pool 생성."""

    pools = {}
    offset = 0
    for organ_id in range(1, 10):
        for type_index, error_type in enumerate(ERROR_TYPES, start=1):
            count = 32 * type_index
            values = np.arange(offset, offset + count, dtype=np.int32)
            pools[(organ_id, error_type)] = np.stack(
                (values, values + 1, values + 2), axis=1
            )
            offset += count + 10
    return ErrorCandidatePools(pools=pools, organ_ids=tuple(range(1, 10)))


def benchmark(
    name: str,
    draws: int,
    selection: Callable[[np.random.Generator], object],
    seed: int,
) -> dict[str, float | int | str]:
    """Candidate selection의 steady CPU wall time 측정."""

    generator = np.random.default_rng(seed)
    for _ in range(100):
        if selection(generator) is None:
            raise AssertionError(f"{name} warm-up에서 None 반환")
    started = time.perf_counter()
    for _ in range(draws):
        if selection(generator) is None:
            raise AssertionError(f"{name} benchmark에서 None 반환")
    seconds = time.perf_counter() - started
    return {
        "method": name,
        "draws": draws,
        "seconds": seconds,
        "microseconds_per_draw": seconds / draws * 1e6,
    }


def network_state_sha256(network: torch.nn.Module) -> str:
    """Parameter·buffer 이름/Shape/dtype/value의 deterministic SHA-256 계산."""

    digest = hashlib.sha256()
    for name, tensor in sorted(network.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    """Trainer common path·frozen protocol·selection cost 검증."""

    arguments = parse_arguments()
    if arguments.draws_per_method < 1:
        raise ValueError("draws-per-method는 positive integer 필요")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(FROZEN_MANIFEST.read_text())
    train_ids = tuple(str(value) for value in manifest["splits"]["train"])
    validation_ids = set(str(value) for value in manifest["splits"]["val"])
    test_ids = set(str(value) for value in manifest["splits"]["test"])
    if len(train_ids) != 525 or set(train_ids) & (validation_ids | test_ids):
        raise AssertionError("Frozen train 525 또는 split disjointness 실패")

    with TemporaryDirectory(prefix="oles3d-protocol-audit-") as temporary:
        os.environ["nnUNet_raw"] = str(NNUNET_ROOT / "nnUNet_raw")
        os.environ["nnUNet_preprocessed"] = str(NNUNET_ROOT / "nnUNet_preprocessed")
        os.environ["nnUNet_results"] = str(Path(temporary) / "results")
        os.environ["OLES3D_RUN_SEED"] = str(arguments.seed)
        os.environ["OLES3D_CANDIDATE_POOL_ROOT"] = str(
            Path(temporary) / "candidate_pools"
        )
        os.environ["OLES3D_OBSERVATIONS_PER_EPOCH"] = "10"
        os.environ["OLES3D_BOUNDARY_TOLERANCE_MM"] = "1.5"
        os.environ["OLES3D_RESERVOIR_CAP"] = "512"
        os.environ["OLES3D_ORGAN_EMA_DECAY"] = "0.9"
        os.environ["OLES3D_ADAPTIVE_FRACTION"] = "0.5"
        os.environ["OLES3D_ERROR_TYPE_EMA_DECAY"] = "0.9"
        os.environ["OLES3D_ERROR_TYPE_ADAPTIVE_FRACTION"] = "0.5"

        from trainers.nnUNetTrainerOLES3DA1Adaptive import (
            nnUNetTrainerOLES3DA1Adaptive,
        )
        from trainers.nnUNetTrainerOLES3DB0Main import nnUNetTrainerOLES3DB0Main
        from trainers.nnUNetTrainerOLES3DB1Static import (
            nnUNetTrainerOLES3DB1Static,
        )
        from trainers.nnUNetTrainerOLES3DPAdaptive import (
            nnUNetTrainerOLES3DPAdaptive,
        )

        plans = json.loads((PREPROCESSED_ROOT / "nnUNetPlans.json").read_text())
        plans["continue_training"] = False
        dataset_json = json.loads(
            (PREPROCESSED_ROOT / "dataset.json").read_text()
        )
        trainer_classes = {
            "B0": nnUNetTrainerOLES3DB0Main,
            "B1": nnUNetTrainerOLES3DB1Static,
            "A1": nnUNetTrainerOLES3DA1Adaptive,
            "P": nnUNetTrainerOLES3DPAdaptive,
        }
        trainers = {
            name: trainer_class(
                plans=dict(plans),
                configuration="3d_fullres",
                fold="all",
                dataset_json=dict(dataset_json),
                device=torch.device("cpu"),
            )
            for name, trainer_class in trainer_classes.items()
        }

        common_fields = {
            name: {
                "batch_size": int(trainer.configuration_manager.batch_size),
                "patch_size_zyx": [
                    int(value)
                    for value in trainer.configuration_manager.patch_size
                ],
                "updates_per_epoch": trainer.num_iterations_per_epoch,
                "schedule_horizon_updates": trainer.schedule_horizon_updates,
                "oversample_foreground_percent": trainer.oversample_foreground_percent,
            }
            for name, trainer in trainers.items()
        }
        if len({json.dumps(value, sort_keys=True) for value in common_fields.values()}) != 1:
            raise AssertionError(f"Comparator common field 불일치: {common_fields}")

        guided_fields = {
            name: {
                "observations_per_epoch": trainer.observations_per_epoch,
                "boundary_tolerance_mm": trainer.boundary_tolerance_mm,
                "reservoir_cap": trainer.reservoir_cap,
            }
            for name, trainer in trainers.items()
            if name != "B0"
        }
        if len({json.dumps(value, sort_keys=True) for value in guided_fields.values()}) != 1:
            raise AssertionError(f"Guided observer field 불일치: {guided_fields}")

        if not (
            nnUNetTrainerOLES3DA1Adaptive.refresh_online_candidates
            is nnUNetTrainerOLES3DB1Static.refresh_online_candidates
            and nnUNetTrainerOLES3DPAdaptive.refresh_online_candidates
            is nnUNetTrainerOLES3DB1Static.refresh_online_candidates
            and nnUNetTrainerOLES3DA1Adaptive.get_dataloaders
            is nnUNetTrainerOLES3DB1Static.get_dataloaders
            and nnUNetTrainerOLES3DPAdaptive.get_dataloaders
            is nnUNetTrainerOLES3DB1Static.get_dataloaders
        ):
            raise AssertionError("B1/A1/P common observer/dataloader path 불일치")
        loader_classes = {
            name: trainers[name]._training_loader_class().__name__
            for name in ("B1", "A1", "P")
        }
        initial_weight_hashes = {}
        for name, trainer in trainers.items():
            random.seed(arguments.seed)
            np.random.seed(arguments.seed)
            torch.manual_seed(arguments.seed)
            trainer.initialize()
            initial_weight_hashes[name] = network_state_sha256(trainer.network)
        if len(set(initial_weight_hashes.values())) != 1:
            raise AssertionError(
                f"Comparator initial weight hash 불일치: {initial_weight_hashes}"
            )

    pools = synthetic_pools()
    organ_state = OrganLearningState(pools.organ_ids, 0.9, 0.5)
    organ_state.update(
        {organ_id: 0.5 + organ_id * 0.04 for organ_id in pools.organ_ids}
    )
    type_state = ErrorTypeLearningState(pools.organ_ids, 0.9, 0.5)
    for organ_id in pools.organ_ids:
        type_state.update(
            organ_id,
            {
                "interior_miss": organ_id,
                "boundary_disagreement": 10 - organ_id,
                "exterior_false_positive": 5,
            },
        )
    benchmarks = [
        benchmark(
            "B1",
            arguments.draws_per_method,
            lambda generator: choose_b1_static_candidate(pools, generator),
            arguments.seed,
        ),
        benchmark(
            "A1",
            arguments.draws_per_method,
            lambda generator: choose_a1_adaptive_candidate(
                pools, organ_state, generator
            ),
            arguments.seed,
        ),
        benchmark(
            "P",
            arguments.draws_per_method,
            lambda generator: choose_p_adaptive_candidate(
                pools, organ_state, type_state, generator
            ),
            arguments.seed,
        ),
    ]
    if any(float(item["microseconds_per_draw"]) > 1_000 for item in benchmarks):
        raise AssertionError("Allocation selection CPU cost가 1 ms/draw 초과")

    result = {
        "schema_version": 1,
        "scope": "comparator_protocol_fairness_and_allocation_cpu_cost",
        "frozen_train_case_count": len(train_ids),
        "split_disjoint": True,
        "common_training_fields": common_fields,
        "common_guided_observer_fields": guided_fields,
        "shared_b1_a1_p_observer_and_dataloader_implementation": True,
        "loader_classes": loader_classes,
        "initial_weight_sha256": initial_weight_hashes,
        "identical_initial_weights": True,
        "authorized_differences": {
            "B0": "nnU-Net default foreground oversampling",
            "B1": "online error pools plus fixed-uniform organ allocation",
            "A1": "B1 plus organ Dice-deficit adaptive allocation",
            "P": "A1 plus organ-local error-type adaptive allocation",
        },
        "allocation_cpu_benchmark": benchmarks,
        "full_training_or_accuracy_measured": False,
        "status": "PASS",
    }
    arguments.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("=== Comparator Protocol Fairness Audit ===")
    print("Frozen train / split:      ", len(train_ids), "/ PASS")
    print("Common training fields:    ", next(iter(common_fields.values())))
    print("Common guided observer:    ", next(iter(guided_fields.values())))
    print("Policy loader classes:     ", loader_classes)
    print("Initial weight SHA-256:    ", next(iter(initial_weight_hashes.values())))
    print("Allocation CPU cost:")
    for item in benchmarks:
        print(
            f"  {item['method']}: {item['microseconds_per_draw']:.3f} us/draw"
        )
    print("Full training measured:     NO")
    print("JSON:                      ", arguments.output)


if __name__ == "__main__":
    main()
