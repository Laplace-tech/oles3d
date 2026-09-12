"""회귀 테스트와 실제 small 감사의 순차 재실행·기존 수치 비교."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from audit_selected_nonselected_collisions import verified_case_directories

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data/raw/totalsegmentator/v2.0.1/small"


def sha256(path: Path) -> str:
    """기존 근거와 수정 코드의 파일 fingerprint 계산."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def old_value_differences(old: Any, new: Any, path: str = "") -> list[str]:
    """추가된 검증 필드를 제외하고 기존 결과 전체의 값 비교."""
    if isinstance(old, dict) and isinstance(new, dict):
        differences: list[str] = []
        for key, value in old.items():
            # 이전의 설명 문자열을 명시적 manifest 목록으로 바꾼 schema 변경
            if path == "" and key == "nonselected_organs":
                continue
            child = f"{path}/{key}"
            if key not in new:
                differences.append(child + " missing")
            else:
                differences.extend(old_value_differences(value, new[key], child))
        return differences
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return [path + " length mismatch"]
        return [difference for index, (left, right) in enumerate(zip(old, new))
                for difference in old_value_differences(left, right, f"{path}/{index}")]
    return [] if old == new else [f"{path}: {old!r} != {new!r}"]


def run_step(name: str, command: list[str], output_dir: Path,
             expected_exit: int = 0) -> dict[str, Any]:
    """명령을 하나씩 실행하고 stdout/stderr·종료 상태 저장."""
    started = time.monotonic()
    print(f"\nSTART {name}", flush=True)
    log_path = output_dir / f"{name}.txt"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        if process.stdout is None:
            raise RuntimeError("stdout pipe 생성 실패")
        for line in process.stdout:
            log.write(line)
            log.flush()
            if line.startswith("[") or "Total geometry failures:" in line:
                print(line.rstrip(), flush=True)
        returncode = process.wait()
    result = {"command": command, "returncode": returncode,
              "expected_exit": expected_exit,
              "passed": returncode == expected_exit,
              "seconds": round(time.monotonic() - started, 3),
              "log": str(log_path)}
    print(f"END {name}: exit={returncode}, expected={expected_exit}, "
          f"seconds={result['seconds']}", flush=True)
    return result


def main() -> int:
    """기존 결과 보존 상태에서 실제 data와 실패 경로 검증."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--executor", choices=("user", "agent", "unknown"), default="unknown",
                        help="실제 실행 주체 기록; 미지정 시 unknown")
    args = parser.parse_args()
    if args.output_dir is None:
        parent = ROOT / "artifacts/data_foundation"
        parent.mkdir(parents=True, exist_ok=True)
        output_dir = Path(tempfile.mkdtemp(prefix="repair-verification-", dir=parent))
    else:
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
    print("Evidence directory:", output_dir, flush=True)

    historical = {
        "full-117": ROOT / "artifacts/data_foundation/1_6c_selected_nonselected.json",
        "organ-part-24": ROOT / "artifacts/data_foundation/1_6d_organ_part_policy.json",
    }
    originals = {str(path): sha256(path) for path in historical.values()}
    code_paths = [Path(__file__), ROOT / "research/data_audit/v201_mask_manifest.py",
                  ROOT / "research/data_audit/audit_selected_nonselected_collisions.py",
                  ROOT / "research/data_foundation/04_audit_geometry.py"]
    report: dict[str, Any] = {"executor": args.executor, "dataset_root": str(DATASET),
                              "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in code_paths},
                              "original_sha256": originals, "steps": [], "comparisons": {}}
    python = sys.executable
    geometry = str(ROOT / "research/data_foundation/04_audit_geometry.py")
    commands = [
        ("geometry_all", [python, geometry, "--dataset-root", str(DATASET), "--cases",
                          *[p.name for p in verified_case_directories(DATASET)]], 0),
        ("geometry_expected_failure", [python, geometry, "--dataset-root", str(DATASET),
                                       "--cases", "s1389", "--organs", "liver", "--tolerance-mm", "0"], 1),
    ]
    try:
        for name, command, expected in commands:
            step = run_step(name, command, output_dir, expected)
            report["steps"].append(step)
            if not step["passed"]:
                raise RuntimeError(f"검증 실패: {name}")
        for scope, previous_path in historical.items():
            destination = output_dir / f"{scope}.json"
            command = [python, str(ROOT / "research/data_audit/audit_selected_nonselected_collisions.py"),
                       "--dataset-root", str(DATASET), "--nonselected-scope", scope,
                       "--output", str(destination)]
            step = run_step(scope, command, output_dir)
            report["steps"].append(step)
            if not step["passed"]:
                raise RuntimeError(f"실제 감사 실패: {scope}")
            previous = json.loads(previous_path.read_text())
            current = json.loads(destination.read_text())
            differences = old_value_differences(previous, current)
            report["comparisons"][scope] = {"differences": differences,
                "case_count": current["case_count"],
                "audit_complete": current["audit_complete"],
                "affected_voxels": current["total_affected_voxels"],
                "affected_fraction": current["affected_fraction_of_selected_foreground"],
                "value_anomaly_count": current["value_anomaly_count"]}
            if differences or not current["audit_complete"]:
                raise RuntimeError(f"이전 수치와 차이 또는 불완전한 감사: {scope}")
            print(f"MATCH {scope}: all historical values unchanged", flush=True)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        print(report["error"], flush=True)
    finally:
        report["originals_unchanged"] = all(sha256(Path(p)) == digest for p, digest in originals.items())
        report["passed"] = "error" not in report and report["originals_unchanged"]
        (output_dir / "verification.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Verification passed:", report["passed"], flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
