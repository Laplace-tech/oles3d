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
- 선택 장기 mask 사이의 voxel overlap과 충돌 장기 pair
- 9개 장기가 모두 존재하는 case 수와 split

`Affine warnings`는 affine 원소가 `1e-4`보다 다르다는 뜻이다. 실제 정렬
실패 판정은 volume의 8개 모서리에서 측정한 최대 물리 좌표 차이가
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

Overlap이 발견된 case는 다음 명령으로 pair별 overlap 깊이, 연결 성분,
bounding box와 CT 대표 단면을 검사한다. 기본 case는 전체 감사에서 overlap
voxel이 가장 많았던 `s0999`다.

```bash
.venv/bin/python research/data_audit/inspect_overlap_case.py \
  --case-id s0999 \
  --top-pairs 3
```

출력은 Git에서 제외되는
`artifacts/data_audit/overlap_inspection/<case-id>/`에 저장된다. PNG 색상은
첫 장기 `blue`, 둘째 장기 `orange`, 실제 overlap `magenta`다. `Deep
overlap`은 두 mask를 각각 6-connected 1 voxel erosion한 뒤에도 겹치는
voxel이다. 값이 0이면 얇은 접촉 경계와 일치하지만, annotation 원인이나
해부학적으로 옳은 class를 자동 확정하지는 않는다.

## 판정 해석

- `Missing files`, `Load errors`, `Geometry errors`,
  `Invalid binary masks`가 모두 0이면 data integrity `PASS`
- `Affine warnings`만 존재하고 검사한 physical displacement가 허용값보다
  작으면 해당 tolerance 내 header 차이로 기록; 원인을 반올림으로 단정하지 않음
- `Non-empty organ coverage`는 해당 mask에 foreground가 있는 case 수이며
  장기 전체의 scan coverage나 annotation 정확성까지 보증하지 않음
- empty mask는 흉부·목 등 제한된 scan coverage에서 발생할 수 있으므로
  파일 손상과 분리하여 cohort eligibility에 사용
- mask overlap이 있으면 multiclass merge gate를 `REVIEW REQUIRED`로 표시

## Overlap 관찰과 변환 후보 — 아직 동결하지 않음

2026-09-12 기준 small subset의 38개 case, 41,336 voxel에서 선택 장기
overlap을 확인했다. 상위 case의 overlap은 1-voxel erosion 뒤에도
50% 이상 남는 사례를 포함한다. 이는 6방향 1-voxel erosion으로 제거되지
않는 배열상 충돌이며, 그것만으로 annotation 생성 과정의 원인을 확정하지
않는다. 두 mask를 1회 erosion한 교집합의 크기를 측정한 것이며, 최대
침범 깊이를 mm로 측정한 결과도 아니다.

TotalSegmentator `v2.0.1`의 공식 `combine_masks_to_multilabel_file`은
`class_map["total"]` 순서로 binary mask를 읽고, foreground voxel에 현재
class ID를 대입한다. 따라서 나중 class가 overlap voxel을 덮어쓴다.
OLES3D의 9개 장기는 해당 v2 class ID 1–9와 순서가 같다. 다만 이 함수의
존재는 배포 dataset의 실제 학습 target이 이 방식으로 만들어졌다는 증거와
다르다. 소프트웨어 tag와 dataset version이 같은 문자열이라는 사실만으로
둘의 생성 이력이 연결되는 것도 아니다.

```text
1 spleen                 6 stomach
2 kidney_right           7 pancreas
3 kidney_left            8 adrenal_gland_right
4 gallbladder            9 adrenal_gland_left
5 liver
```

예를 들어 `gallbladder(4) + liver(5)`는 `liver(5)`,
`spleen(1) + stomach(6)`는 `stomach(6)`으로 변환한다. 이 우선순위는
해부학적으로 최적인 class를 보증하지 않는다. 또한 selected-nine merge와
full-class merge 후 background remap은 동등하다고 가정할 수 없다.
예를 들어 같은 voxel에 liver(5)와 비선택 class(10 이상)가 존재하면 전자는
liver를 남기지만 후자는 비선택 class로 overwrite한 뒤 background가 된다.
현재 nine-organ overlap 감사는 이 selected/nonselected 충돌을 검사하지 않았다.

2026-09-12 hands-on review에서 기존 corner displacement 함수가 8개 중
5개 corner만 검사하는 오류를 발견했다. 8개 전체 조합으로 수정한 결과
`s1389`의 최대값은 `[0, 215, 89]` corner에서 `0.062126 mm`였으며,
기존 `0.056096 mm`보다 컸다. `0.1 mm` tolerance 이내이므로 geometry
판정은 그대로 PASS다.

이전의 확정 표현은 2026-09-12 정책 검토에서 PROPOSED로 정정했다.
진행 상태와 다음 검증 범위는 [연구 checkpoint](../README.md)에 기록한다.
`inspect_overlap_case.py`의 `Official v2 winner`는 두 장기 사이의 순서에
따른 승자를 뜻하며, 모든 class를 합친 최종 voxel label이나 정답 판정이 아니다.

Selected/nonselected 충돌 감사 중 배포 archive의
`s0726/segmentations/costal_cartilages.nii.gz`에서 `{0, 1}` 이외 값 `4`가
6 voxels 발견됐다. Extracted file과 ZIP 내부 원본의 SHA-256가 일치하고
NIfTI scaling도 `1.0/0.0`이므로 extraction 또는 scaling 문제는 아니다.
공식 v2.0.1 결합 helper가 `img > 0.5`를 foreground로 사용하므로 충돌
감사도 같은 판정을 사용하되, non-binary value를 JSON anomaly로 별도
기록한다. 이는 값 `4`의 annotation 생성 원인을 확정하거나 원본을
수정한다는 뜻이 아니다.

전체 102-case 재실행은 24분 25초가 걸렸고 `audit_complete=true`, skipped
mask 0으로 끝났다. Selected-nine merge와 full-117 merge 후 background
remap은 96 cases, 177,133 unique voxels에서 달랐으며 selected
foreground의 0.276568%다. 가장 많이 영향을 받은 선택 장기는 stomach,
liver, pancreas 순이었다. 값 4 anomaly는 s0726의 6 voxels 외에 s0928의
1 voxel에서도 확인됐다.

이 수치는 full-117 정책과 selected-nine 정책의 비동등성을 증명하지만
full-117 정책을 채택해야 한다는 뜻은 아니다. V2 class map은 117개 구조를
organ, vertebrae, cardiac, muscle, rib의 다섯 part로 나누며 선택 9장기는
24-class organ part에 속한다. 다음 정책 gate는 같은 organ part의 나머지
15개 구조로 범위를 제한한 비교다.

```bash
.venv/bin/python \
  research/data_audit/audit_selected_nonselected_collisions.py \
  --case-ids s0726 \
  --output artifacts/data_audit/s0726_selected_nonselected.json
```

- [TotalSegmentator v2.0.1 label map](https://github.com/wasserth/TotalSegmentator/blob/v2.0.1/totalsegmentator/map_to_binary.py)
- [TotalSegmentator v2.0.1 multilabel 변환 함수](https://github.com/wasserth/TotalSegmentator/blob/v2.0.1/totalsegmentator/libs.py#L273-L292)
