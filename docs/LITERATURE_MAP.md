# OLES3D Literature Map

상태: seed map v0.1

기준일: 2026-09-04

## 1. Bottom line

초기 가제였던 “장기별 예측 오류를 추적해 patch를 adaptive sampling한다”는 문장만으로는 novelty가 없다.

- 2017 adaptive sampling은 training 중 posterior error map으로 difficult region에 집중했다.
- 2022 OHPM은 shape model과 multi-armed bandit으로 online hard patch mining을 했다.
- 2026 APS는 model-state prediction error와 uncertainty로 voxel-wise sampling distribution을 만들었다.

따라서 OLES3D는 **dynamic error sampling 자체**를 claim하지 않는다. 검증할 candidate contribution은 `low-dimensional organ × error-type learning state`, `label-derived static candidate pools`, `no extra full-volume error map`, `default-branch telemetry`, `single 8 GB GPU에서의 measured trade-off`의 결합이다.

## 2. Core foundation

| Work | What it establishes | OLES3D consequence |
|---|---|---|
| nnU-Net, Nature Methods 2021 | dataset fingerprint와 self-configuring segmentation pipeline | architecture가 아니라 sampler만 바꾸며 unmodified-sampler nnU-Net을 reference로 둔다. |
| nnU-Net Revisited, 2024 | 3D segmentation에서 rigorous validation과 fair compute comparison의 중요성 | 새 architecture 비교를 삭제하고 repeated/equal-budget protocol을 둔다. |
| TotalSegmentator, Radiology: AI 2023 | original broad CT multi-anatomy dataset/system | 별도 v2.0.1 dataset record의 1,228 CT에서 좁은 9-organ task를 만든다. v1 paper와 v2 dataset을 같은 cohort로 서술하지 않고 production model superiority도 주장하지 않는다. |

Primary links:

- nnU-Net paper: <https://www.nature.com/articles/s41592-020-01008-z>
- nnU-Net official repository: <https://github.com/MIC-DKFZ/nnUNet>
- PyTorch 2.9.0 Conv3D AMP regression issue: <https://github.com/pytorch/pytorch/issues/166122>
- nnU-Net Revisited: <https://arxiv.org/abs/2404.09556>
- TotalSegmentator paper: <https://pubs.rsna.org/doi/10.1148/ryai.230024>
- TotalSegmentator v2 dataset: <https://zenodo.org/records/10047292>
- TotalSegmentator official repository: <https://github.com/wasserth/TotalSegmentator>
- Official retraining guide: <https://github.com/wasserth/TotalSegmentator/blob/master/resources/train_nnunet.md>

## 3. Closest sampling prior

| Work | Signal | Spatial representation | Update mechanism | Extra machinery | Collision with initial idea | Planned distinction |
|---|---|---|---|---|---|---|
| Berger et al., 2017, Adaptive Sampling (arXiv preprint) | posterior prediction error | dense error map | throughout training | dual-path CNN in full method | direct collision with “sample current errors” | no dense map; organ/type state + static pools |
| He et al., 2022, OHPM | predicted hard-patch feedback | registered average shape space | online bandit | shape model + registration + bandit | direct collision with online hard multi-organ patches | no atlas/registration/bandit; transparent deterministic controller |
| Park et al., 2026, APS+PE | uncertainty + prediction error | voxel-wise guidance map/patch graph | model-state adaptive random walk | PE attention block | nearly exact collision with generic dynamic model-error sampling | no uncertainty/map/random walk/PE; error-type state and label pools only |
| nnU-Net default | foreground presence | class voxel locations | fixed | self-configured pipeline | strong baseline already selects foreground/class voxels | retain as 50% safety branch and measure added value |

Primary links:

- Adaptive Sampling, 2017: <https://arxiv.org/abs/1709.02764>
- OHPM, IEEE JBHI 2022: <https://pubmed.ncbi.nlm.nih.gov/34928809/>
- APS+PE, Scientific Reports 2026: <https://pmc.ncbi.nlm.nih.gov/articles/PMC13377078/>
- nnU-Net data loader source: <https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/dataloading/data_loader.py>

## 4. Adjacent work that can invalidate a claim

다음 topic도 Gate 0 search에 포함한다.

- class-balanced and size-aware patch sampling
- boundary-focused patch/crop sampling
- hard-negative mining for segmentation
- false-positive/false-negative adaptive loss
- curriculum sampling based on class learning progress
- online batch selection and loss-based sampling
- spatial priors/atlas-guided multi-organ sampling
- dynamic data selection under fixed compute

특히 adaptive region-specific loss가 FP/FN penalty를 조정하는 연구가 있으므로, OLES3D의 claim은 “error type을 처음 사용했다”가 아니다. **loss가 아니라 future patch allocation을 바꾸며, low-dimensional state와 static spatial pools로 구현했다**가 검증 대상이다.

Reference:

- Adaptive Region-Specific Loss: <https://pubmed.ncbi.nlm.nih.gov/37363838/>
- Boundary-sensitive liver patch sampling example: <https://pubmed.ncbi.nlm.nih.gov/29633960/>
- Class imbalance in head-and-neck nnU-Net: <https://pubmed.ncbi.nlm.nih.gov/35578086/>

## 5. Novelty Gate 0 search protocol

### Databases

- PubMed
- IEEE Xplore
- arXiv
- Google Scholar or Semantic Scholar for citation chaining
- paper reference lists and citing papers

### Query families

```text
(3D OR volumetric) medical segmentation adaptive patch sampling
multi-organ segmentation online hard patch mining
organ-aware class-aware dynamic sampling segmentation
false positive false negative boundary error patch sampling
learning progress curriculum sampling medical segmentation
nnU-Net custom foreground oversampling boundary sampling
resource constrained 3D segmentation sampling efficiency
```

### Inclusion

- segmentation training sample/patch/location selection
- medical 2D/3D or directly transferable dense segmentation
- class/organ-aware, error-aware, boundary-aware, curriculum or hard mining
- peer-reviewed paper or clearly identified preprint with methods detail

### Extraction fields

- citation/DOI/version/date
- task and dataset
- baseline
- sampling unit: case/slice/patch/voxel
- signal: label size/loss/error/uncertainty/progress
- spatial representation
- update interval
- additional inference/model/storage
- fairness budget
- metrics and seeds
- released code/license
- exact claim
- overlap with OLES3D

### Stop rule

Backward and forward citation chaining stops when two consecutive search rounds add no mechanism that changes the claim matrix. “검색 결과가 없다”가 아니라 query, date와 screened title을 log로 남긴다.

## 6. Claim ladder

논문 표현은 evidence에 따라 한 단계씩만 올라간다.

1. **Proposal:** We propose an organ/error-type learning-state sampler.
2. **Difference:** Unlike closest tested approaches, it uses low-dimensional feedback and static pools.
3. **Efficiency:** It avoids extra full-volume error-map inference and measured overhead is X%.
4. **Effect:** It improves the predefined endpoint under the fixed protocol.
5. **Generalization:** 금지 unless another dataset/model is tested after core completion.

“first”, “novel”, “state of the art”, “clinical utility”는 comprehensive evidence 없이는 쓰지 않는다.

## 7. Reading order for 마벨러스

1. nnU-Net paper and current data-loader source
2. TotalSegmentator paper/data/retraining guide
3. Berger 2017 adaptive sampling
4. OHPM 2022
5. APS+PE 2026
6. nnU-Net Revisited
7. adjacent boundary/class/error works found by Gate 0

각 논문은 abstract 요약으로 끝내지 않고 `problem -> signal -> sampling unit -> feedback -> cost -> comparison -> limitation` 일곱 칸으로 한 장에 정리한다.
