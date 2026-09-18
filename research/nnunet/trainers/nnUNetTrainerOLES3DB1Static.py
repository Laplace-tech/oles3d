from __future__ import annotations

import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
    NonDetMultiThreadedAugmenter,
)
from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)
from nnunetv2.configuration import get_allowed_n_proc_DA
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class


RESEARCH_ROOT = Path(__file__).resolve().parents[2]
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from sampling.candidate_pool_store import (
    CandidatePoolStore,
    save_case_candidate_pools,
)
from sampling.nnunet_guided_loader import (
    GuidedCandidateDataset,
    nnUNetDataLoaderOLES3DB1,
)
from sampling.organ_allocation_store import OrganAllocationStore
from sampling.organ_learning_state import measure_hard_dice_by_organ
from sampling.p_allocation_store import PAllocationStore
from sampling.observer_schedule import ObserverAssignment, ObserverSchedule
from sampling.online_observer import (
    make_empty_candidate_pools,
    observe_patch_errors,
    refresh_candidate_reservoir,
    select_observer_patch,
)

try:
    # `trainers.*` package로 import되는 local runner 경로
    from .nnUNetTrainerOLES3DB0Main import nnUNetTrainerOLES3DB0Main
except ImportError:
    # nnU-Net external-trainer discovery의 top-level module 경로
    from nnUNetTrainerOLES3DB0Main import nnUNetTrainerOLES3DB0Main


class nnUNetTrainerOLES3DB1Static(nnUNetTrainerOLES3DB0Main):
    """동일 force-foreground slot에 B1 static candidate를 사용하는 trainer."""

    def __init__(
        self,
        plans: dict[str, Any],
        configuration: str,
        fold: int | str,
        dataset_json: dict[str, Any],
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            device=device,
        )
        self.observations_per_epoch = int(
            os.environ.get("OLES3D_OBSERVATIONS_PER_EPOCH", "10")
        )
        self.boundary_tolerance_mm = float(
            os.environ.get("OLES3D_BOUNDARY_TOLERANCE_MM", "1.5")
        )
        self.reservoir_cap = int(
            os.environ.get("OLES3D_RESERVOIR_CAP", "512")
        )
        if self.observations_per_epoch < 1 or self.reservoir_cap < 1:
            raise ValueError("Observer count와 reservoir cap은 양수 필요")
        self._observer_dataset: Any | None = None
        self._candidate_pool_store: CandidatePoolStore | None = None
        self._observer_schedule: ObserverSchedule | None = None

    def _allocation_store(self) -> OrganAllocationStore | None:
        """B1에는 adaptive allocation state 없음."""

        return None

    def _training_loader_class(self) -> type[nnUNetDataLoaderOLES3DB1]:
        """현재 policy의 guided training loader class 반환."""

        return nnUNetDataLoaderOLES3DB1

    def _p_allocation_store(self) -> PAllocationStore | None:
        """B1/A1에는 joint organ×type allocation state 없음."""

        return None

    def _augment_observer_report(self, report: dict[str, Any]) -> None:
        """Subclass별 observer state 갱신 hook."""

    def _extra_candidate_state_paths(self) -> tuple[Path, ...]:
        """Candidate archive에 함께 저장할 추가 state 경로 반환."""

        return ()

    def _candidate_pool_root(self) -> Path:
        """환경변수로 지정된 case snapshot directory 확인."""

        value = os.environ.get("OLES3D_CANDIDATE_POOL_ROOT")
        if not value:
            raise RuntimeError("OLES3D_CANDIDATE_POOL_ROOT 환경변수 필요")
        root = Path(value).resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root

    def get_dataloaders(self) -> tuple[Any, Any]:
        """B1 train loader와 변경 없는 nnU-Net validation loader 생성."""

        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(
                self.preprocessed_dataset_folder
            )

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        (
            rotation_for_data_augmentation,
            do_dummy_2d_data_augmentation,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        training_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_data_augmentation,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_augmentation,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=(
                self.label_manager.foreground_regions
                if self.label_manager.has_regions
                else None
            ),
            ignore_label=self.label_manager.ignore_label,
        )
        validation_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=(
                self.label_manager.foreground_regions
                if self.label_manager.has_regions
                else None
            ),
            ignore_label=self.label_manager.ignore_label,
        )

        base_training_dataset, validation_dataset = (
            self.get_tr_and_val_datasets()
        )
        self._observer_dataset = base_training_dataset
        self._candidate_pool_store = CandidatePoolStore(
            self._candidate_pool_root()
        )
        self._observer_schedule = ObserverSchedule(
            case_ids=tuple(str(value) for value in base_training_dataset.identifiers),
            epochs=self.num_epochs,
            observations_per_epoch=self.observations_per_epoch,
            seed=self.run_seed,
        )
        guided_training_dataset = GuidedCandidateDataset(
            base_dataset=base_training_dataset,
            pool_store=self._candidate_pool_store,
            allocation_store=self._allocation_store(),
            p_allocation_store=self._p_allocation_store(),
        )
        training_loader = self._training_loader_class()(
            guided_training_dataset,
            self.batch_size,
            initial_patch_size,
            patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=training_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
            selection_seed=self.run_seed,
        )
        validation_loader = nnUNetDataLoader(
            validation_dataset,
            self.batch_size,
            patch_size,
            patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=validation_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
        )

        allowed_processes = get_allowed_n_proc_DA()
        if allowed_processes == 0:
            augmented_training = SingleThreadedAugmenter(training_loader, None)
            augmented_validation = SingleThreadedAugmenter(validation_loader, None)
        else:
            augmented_training = NonDetMultiThreadedAugmenter(
                data_loader=training_loader,
                transform=None,
                num_processes=allowed_processes,
                num_cached=max(6, allowed_processes // 2),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
            augmented_validation = NonDetMultiThreadedAugmenter(
                data_loader=validation_loader,
                transform=None,
                num_processes=max(1, allowed_processes // 2),
                num_cached=max(3, allowed_processes // 4),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )

        # nnU-Net 기본 계약과 동일한 worker warm-up
        _ = next(augmented_training)
        _ = next(augmented_validation)
        return augmented_training, augmented_validation

    @torch.inference_mode()
    def refresh_online_candidates(
        self,
        assignments: tuple[ObserverAssignment, ...],
    ) -> dict[str, Any]:
        """Current network로 지정 case snapshot을 atomic refresh."""

        if self._observer_dataset is None or self._candidate_pool_store is None:
            raise RuntimeError("Observer dataset/store가 아직 초기화되지 않음")
        if not assignments:
            raise ValueError("Observer assignment는 하나 이상 필요")

        network_was_training = self.network.training
        self.set_deep_supervision_enabled(False)
        self.network.eval()
        started = time.perf_counter()
        case_reports: list[dict[str, Any]] = []
        try:
            for assignment in assignments:
                case_started = time.perf_counter()
                data, segmentation, _, properties = (
                    self._observer_dataset.load_case(assignment.case_id)
                )
                generator = np.random.default_rng(
                    assignment.observation_seed
                )
                observer_patch = select_observer_patch(
                    data_czyx=np.asarray(data),
                    target_czyx=np.asarray(segmentation),
                    class_locations=properties["class_locations"],
                    focus_organ_id=assignment.focus_organ_id,
                    patch_size_zyx=tuple(
                        int(value)
                        for value in self.configuration_manager.patch_size
                    ),
                    generator=generator,
                )
                input_tensor = torch.from_numpy(
                    observer_patch.data_czyx[None]
                ).to(self.device, dtype=torch.float32)
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.device.type == "cuda",
                    dtype=torch.float16,
                ):
                    logits = self.network(input_tensor)  # [1, K=10, D, H, W]
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                prediction_zyx = (
                    logits.argmax(dim=1)[0].detach().cpu().numpy()
                )  # [D, H, W]

                observation = observe_patch_errors(
                    observer_patch=observer_patch,
                    prediction_zyx=prediction_zyx,
                    spacing_zyx_mm=tuple(
                        float(value) for value in properties["spacing"]
                    ),
                    boundary_tolerance_mm=self.boundary_tolerance_mm,
                    maximum_candidates_per_stratum=self.reservoir_cap,
                    generator=generator,
                )
                focus_dice = measure_hard_dice_by_organ(
                    target_zyx=observer_patch.target_czyx[0],
                    prediction_zyx=prediction_zyx,
                    organ_ids=(assignment.focus_organ_id,),
                )[assignment.focus_organ_id]
                previous = self._candidate_pool_store.load(
                    assignment.case_id
                )
                if previous is None:
                    previous = make_empty_candidate_pools()
                refreshed = refresh_candidate_reservoir(
                    previous_pools=previous,
                    observation=observation,
                    maximum_candidates_per_stratum=self.reservoir_cap,
                    generator=generator,
                )
                save_case_candidate_pools(
                    self._candidate_pool_store.case_path(assignment.case_id),
                    refreshed,
                )
                case_reports.append(
                    {
                        "case_id": assignment.case_id,
                        "focus_organ_id": assignment.focus_organ_id,
                        "visit_cycle": assignment.visit_cycle,
                        "observation_seed": assignment.observation_seed,
                        "focus_organ_dice": focus_dice,
                        "focus_error_counts": {
                            error_type: int(
                                observation.error_counts[
                                    (assignment.focus_organ_id, error_type)
                                ]
                            )
                            for error_type in (
                                "interior_miss",
                                "boundary_disagreement",
                                "exterior_false_positive",
                            )
                        },
                        "bbox_lbs_zyx": list(observation.bbox_lbs_zyx),
                        "bbox_ubs_zyx": list(observation.bbox_ubs_zyx),
                        "raw_error_voxels": int(
                            sum(observation.error_counts.values())
                        ),
                        "saved_candidate_voxels": int(
                            sum(len(value) for value in refreshed.pools.values())
                        ),
                        "seconds": time.perf_counter() - case_started,
                    }
                )
        finally:
            self.set_deep_supervision_enabled(self.enable_deep_supervision)
            self.network.train(network_was_training)

        report = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "epoch_index": assignments[0].epoch_index,
            "completed_epochs": assignments[0].epoch_index + 1,
            "assignment_count": len(assignments),
            "boundary_tolerance_mm": self.boundary_tolerance_mm,
            "reservoir_cap": self.reservoir_cap,
            "cases": case_reports,
            "total_seconds": time.perf_counter() - started,
        }
        self._augment_observer_report(report)
        self._write_observer_evidence(report)
        return report

    def _write_observer_evidence(self, report: dict[str, Any]) -> None:
        """Epoch observer JSONL과 resume-audit state 기록."""

        output_directory = Path(self.output_folder)
        output_directory.mkdir(parents=True, exist_ok=True)
        history_path = output_directory / "oles3d_observer_history.jsonl"
        with history_path.open("a", encoding="utf-8") as history_file:
            history_file.write(
                json.dumps(report, ensure_ascii=False) + "\n"
            )

        state_path = output_directory / "oles3d_observer_state.json"
        temporary_path = output_directory / ".oles3d_observer_state.tmp"
        state = {
            "schema_version": 1,
            "completed_epochs": report["completed_epochs"],
            "last_epoch_index": report["epoch_index"],
            "observations_per_epoch": self.observations_per_epoch,
            "boundary_tolerance_mm": self.boundary_tolerance_mm,
            "reservoir_cap": self.reservoir_cap,
            "run_seed": self.run_seed,
        }
        temporary_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, state_path)

    def _archive_candidate_state(self, completed_epochs: int) -> None:
        """Resume checkpoint와 같은 epoch의 pool state hard-link 보존."""

        pool_root = self._candidate_pool_root()
        output_directory = Path(self.output_folder)
        state_path = output_directory / "oles3d_observer_state.json"
        if not state_path.is_file():
            raise RuntimeError("Checkpoint 저장 전 observer state 없음")
        state = json.loads(state_path.read_text())
        if int(state["completed_epochs"]) != completed_epochs:
            raise RuntimeError(
                "Checkpoint/candidate epoch 불일치: "
                f"checkpoint={completed_epochs}, "
                f"candidate={state['completed_epochs']}"
            )

        archive_root = output_directory / "candidate_state_checkpoints"
        archive_root.mkdir(parents=True, exist_ok=True)
        archive_directory = archive_root / f"epoch_{completed_epochs:03d}"
        if archive_directory.is_dir():
            return
        temporary_directory = archive_root / f".epoch_{completed_epochs:03d}.tmp"
        if temporary_directory.exists():
            shutil.rmtree(temporary_directory)
        temporary_directory.mkdir(parents=True)
        for source_path in sorted(pool_root.glob("*.npz")):
            os.link(source_path, temporary_directory / source_path.name)
        for source_path in self._extra_candidate_state_paths():
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"추가 candidate state 없음: {source_path}"
                )
            shutil.copy2(source_path, temporary_directory / source_path.name)
        shutil.copy2(
            state_path,
            temporary_directory / "oles3d_observer_state.json",
        )
        os.replace(temporary_directory, archive_directory)

    def save_checkpoint(self, filename: str) -> None:
        """Model checkpoint와 matching candidate state archive 저장."""

        super().save_checkpoint(filename)
        checkpoint_name = Path(filename).name
        should_archive = (
            checkpoint_name == "checkpoint_latest.pth"
            or checkpoint_name == "checkpoint_final.pth"
            or (
                checkpoint_name.startswith("checkpoint_")
                and checkpoint_name[11:17].isdigit()
            )
        )
        if should_archive:
            self._archive_candidate_state(self.current_epoch + 1)

    def on_epoch_end(self) -> None:
        """각 epoch 종료 시 공통 schedule의 current-model 후보 갱신."""

        if self._observer_schedule is None:
            raise RuntimeError("Observer schedule이 초기화되지 않음")
        assignments = self._observer_schedule.assignments_for_epoch(
            self.current_epoch
        )
        report = self.refresh_online_candidates(assignments)
        self.print_to_log_file(
            "OLES3D observer refresh: "
            f"epoch={self.current_epoch}, cases={len(assignments)}, "
            f"seconds={report['total_seconds']:.2f}",
            also_print_to_console=True,
        )
        super().on_epoch_end()
