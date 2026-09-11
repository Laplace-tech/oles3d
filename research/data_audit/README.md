# Dataset Audit

TotalSegmentator v2.0.1 small subset의 CT와 OLES3D 대상 9개 장기 mask를
읽기 전용으로 검사한다. 원본 NIfTI와 `meta.csv`는 변경하지 않는다.

## 실행

프로젝트 루트에서 다음 명령을 실행한다.

```bash
.venv/bin/python research/data_audit/audit_small_dataset.py
```

이 명령은 다음 항목을 WSL console에 출력한다.

- metadata와 case directory 대응
- official split과 study type 분포
- CT Shape, spacing, orientation 범위
- CT 및 선택 장기 mask 누락·로딩 오류
- CT-mask Shape와 affine 기반 physical-space 정렬
- binary mask 값 범위와 장기별 non-empty coverage
- 9개 장기가 모두 존재하는 case 수와 split

`Affine warnings`는 affine 원소가 `1e-4`보다 다르다는 뜻이다. 실제 정렬
실패 판정은 volume 모서리의 최대 물리 좌표 차이가
`--corner-tolerance-mm`을 초과했는지로 구분한다. 기본값은 `0.1 mm`이며
명령에 명시해 변경할 수 있다.

```bash
.venv/bin/python research/data_audit/audit_small_dataset.py \
  --corner-tolerance-mm 0.1
```

aggregate 결과가 필요할 때만 Git에서 제외되는 `artifacts/`에 JSON을
저장한다.

```bash
.venv/bin/python research/data_audit/audit_small_dataset.py \
  --json-output artifacts/data_audit/small_v201.json
```

## 판정 해석

- `Missing files`, `Load errors`, `Geometry errors`,
  `Invalid binary masks`가 모두 0이면 구조 감사 `PASS`
- `Affine warnings`만 존재하고 최대 physical displacement가 허용값보다
  작으면 수치 반올림 차이로 기록
- `Non-empty organ coverage`는 scan range에 실제 장기가 들어온 case 수
- empty mask는 흉부·목 등 제한된 scan coverage에서 발생할 수 있으므로
  파일 손상과 분리하여 cohort eligibility에 사용
