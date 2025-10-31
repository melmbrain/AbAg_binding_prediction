# AbAg 결합 친화도 예측

**딥러닝을 활용한 프로덕션급 항체-항원 결합 친화도 예측 패키지**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**📖 Language / 언어:**
- **한국어**: 지금 보고 계십니다! ([README_KR.md](README_KR.md))
- **English**: For English documentation, click [here](README.md)

## 개요

AbAg Affinity는 아미노산 서열로부터 항체-항원 결합 친화도(pKd/Kd)를 예측하는 딥러닝 모델입니다. 7,015개의 실험적으로 검증된 항체-항원 쌍으로 학습되었습니다.

**성능 지표:**
- **Spearman ρ = 0.8501** (순위 상관관계)
- **Pearson r = 0.9461** (선형 상관관계)
- **R² = 0.8779** (결정 계수)

**주요 기능:**
- 결합 친화도 예측을 위한 프로덕션급 API
- Heavy chain 단독 및 Heavy+Light chain 항체 지원
- ESM-2 단백질 언어 모델 임베딩
- 멀티헤드 어텐션 아키텍처
- GPU 가속 (자동 감지)
- 진행률 표시바를 포함한 배치 처리
- 포괄적인 입력 검증 및 오류 처리

## 설치

### 소스에서 설치

```bash
# 저장소 클론
git clone https://github.com/yourusername/AbAg_binding_prediction.git
cd AbAg_binding_prediction

# 의존성 설치
pip install -r requirements.txt

# 패키지 설치
pip install -e .
```

### 요구사항

- Python 3.8+
- PyTorch 1.12+
- Transformers 4.20+
- NumPy, Pandas, Scikit-learn
- tqdm (진행률 표시바)

## 빠른 시작

### 기본 사용법

```python
from abag_affinity import AffinityPredictor

# 예측기 초기화 (모델과 ESM-2를 자동으로 로드)
predictor = AffinityPredictor()

# 결합 친화도 예측
result = predictor.predict(
    antibody_heavy="EVQLQQSGPGLVKPSQTLSLTCAISGDSVSSNSAAWNWIRQSPSRGLEWLGRTYYRSKW...",
    antibody_light="DIQMTQSPSSLSASVGDRVTITCRASQGIRNYLAWYQQKPGKAPKLLIYAASTLQSGVPS...",
    antigen="KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSR..."
)

# 결과 확인
print(f"pKd: {result['pKd']:.2f}")
print(f"Kd: {result['Kd_nM']:.1f} nM")
print(f"카테고리: {result['category']}")
print(f"해석: {result['interpretation']}")
```

**출력:**
```
pKd: 8.52
Kd: 3.0 nM
카테고리: excellent
해석: Very strong binder (sub-nanomolar, Kd ~ 1-10 nM)
```

### Heavy Chain만 사용하는 항체

```python
# Heavy chain만으로도 작동 (예: VHH/나노바디)
result = predictor.predict(
    antibody_heavy="EVQLQQSGPGLVKPSQTLSLTCAISG...",
    antigen="KVFGRCELAAAMKRHGLD..."
)
```

### 배치 처리

```python
# 여러 쌍을 한번에 예측
pairs = [
    {
        'id': 'Ab001',
        'antibody_heavy': 'EVQ...',
        'antibody_light': 'DIQ...',
        'antigen': 'KVF...'
    },
    {
        'id': 'Ab002',
        'antibody_heavy': 'QVQ...',
        'antigen': 'KVF...'
    }
]

results = predictor.predict_batch(pairs, show_progress=True)

# 결과 처리
for result in results:
    if result['status'] == 'success':
        print(f"{result['id']}: pKd = {result['pKd']:.2f}")
```

## API 레퍼런스

### AffinityPredictor

결합 친화도 예측을 위한 메인 클래스입니다.

```python
AffinityPredictor(model_path=None, device=None, verbose=True)
```

**매개변수:**
- `model_path` (str, optional): 모델 파일 경로. None이면 자동 감지
- `device` (str, optional): 사용할 디바이스 ('cuda', 'cpu', 또는 None for 자동 감지)
- `verbose` (bool): 초기화 메시지 출력 여부

### predict()

단일 항체-항원 쌍의 결합 친화도를 예측합니다.

```python
predict(antibody_heavy, antigen, antibody_light="")
```

**매개변수:**
- `antibody_heavy` (str): Heavy chain 아미노산 서열 (필수)
- `antigen` (str): 항원 아미노산 서열 (필수)
- `antibody_light` (str): Light chain 아미노산 서열 (선택)

**반환값:**
다음 키를 포함하는 딕셔너리:
- `pKd` (float): 예측된 pKd 값
- `Kd_M` (float): Kd (몰 단위)
- `Kd_nM` (float): Kd (나노몰 단위)
- `Kd_uM` (float): Kd (마이크로몰 단위)
- `category` (str): 결합 카테고리 ('excellent', 'good', 'moderate', 'poor')
- `interpretation` (str): 사람이 읽을 수 있는 해석

### predict_batch()

여러 쌍의 친화도를 예측합니다.

```python
predict_batch(pairs, show_progress=True)
```

**매개변수:**
- `pairs` (list): 'antibody_heavy', 'antigen' 키를 포함하는 딕셔너리 리스트. 선택적으로 'antibody_light'와 'id' 포함
- `show_progress` (bool): 진행률 표시바 표시 여부 (tqdm 필요)

**반환값:**
추가 'status' 필드를 포함하는 예측 딕셔너리 리스트

## 결과 이해하기

### pKd 값

pKd = -log10(Kd), Kd는 해리 상수입니다.

| pKd 범위 | 카테고리 | Kd 범위 | 해석 |
|---------|---------|---------|------|
| > 10 | 우수 | < 1 nM | 예외적인 결합체 (피코몰 수준) |
| 9-10 | 우수 | 1-10 nM | 매우 강한 결합체 (서브 나노몰) |
| 7.5-9 | 좋음 | 10-100 nM | 강한 결합체 (나노몰) |
| 6-7.5 | 보통 | 0.1-10 μM | 중간 결합체 (마이크로몰) |
| 4-6 | 약함 | 10-100 μM | 약한 결합체 |
| < 4 | 약함 | > 100 μM | 매우 약하거나 비결합체 |

### 결합 카테고리

- **우수 (Excellent)** (pKd > 9): 치료용 품질 항체
- **좋음 (Good)** (pKd 7.5-9): 연구용 품질 항체
- **보통 (Moderate)** (pKd 6-7.5): 약한 친화도, 최적화 필요
- **약함 (Poor)** (pKd < 6): 비특이적 또는 매우 약한 결합

## 모델 아키텍처

**특징 추출:**
- ESM-2 단백질 언어 모델 (facebook/esm2_t12_35M_UR50D)
- 640차원 임베딩을 서열당 150차원으로 축소
- 항체 + 항원 특징 연결 (300차원)

**신경망:**
- 멀티헤드 어텐션 (8개 헤드)
- 피드포워드 네트워크: 300 → 256 → 128 → 1
- 레이어 정규화 및 잔차 연결
- 정규화를 위한 드롭아웃

**학습:**
- 7,015개 항체-항원 쌍
- 다양한 출처: SAbDab, IEDB, PDB, 문헌
- 친화도 범위에 걸친 균형 잡힌 데이터셋
- 교차 검증된 성능

## 성능 검증

**테스트 세트 성능:**
- Spearman ρ = 0.8501 (순위 정확도)
- Pearson r = 0.9461 (선형 상관관계)
- R² = 0.8779 (설명된 분산)
- MAE = 0.45 pKd 단위

**견고성:**
- 친화도 범위 전반에 걸쳐 작동 (pKd 4-12)
- 다양한 항원 유형 처리
- IgG 및 VHH 형식 모두 지원

## 예제

### 예제 1: 단일 예측

```python
from abag_affinity import AffinityPredictor

predictor = AffinityPredictor()

# 치료용 항체 후보
result = predictor.predict(
    antibody_heavy="EVQLQQSGPGLVKPSQTLSLTCAISGDSVSSNSAAWNWIRQSPSRGLEWLG...",
    antibody_light="DIQMTQSPSSLSASVGDRVTITCRASQGIRNYLAWYQQKPGKAPKLLIYA...",
    antigen="KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYG..."
)

print(f"결합 친화도: {result['pKd']:.2f} (Kd = {result['Kd_nM']:.1f} nM)")
print(f"카테고리: {result['category']}")
```

### 예제 2: 항체 라이브러리 스크리닝

```python
import pandas as pd

# 항체 라이브러리 로드
library = pd.read_csv('antibody_library.csv')

# 쌍 준비
pairs = [
    {
        'id': row['antibody_id'],
        'antibody_heavy': row['heavy_chain'],
        'antibody_light': row['light_chain'],
        'antigen': target_antigen
    }
    for _, row in library.iterrows()
]

# 배치 예측
results = predictor.predict_batch(pairs)

# 친화도로 순위 매기기
df_results = pd.DataFrame(results)
df_ranked = df_results.sort_values('pKd', ascending=False)

# 상위 10개 후보
print(df_ranked[['id', 'pKd', 'Kd_nM', 'category']].head(10))
```

### 예제 3: 변이체 비교

```python
# 항체 변이체 비교
variants = {
    'Original': 'EVQ...',
    'Mutant_Y32F': 'EVQ...F...',
    'Mutant_S52A': 'EVQ...A...'
}

for name, heavy_chain in variants.items():
    result = predictor.predict(
        antibody_heavy=heavy_chain,
        antigen=target_antigen
    )
    print(f"{name}: pKd = {result['pKd']:.2f}")
```

## 문제 해결

### 일반적인 문제

**"Model not found" (모델을 찾을 수 없음)**
- 모델 파일이 `models/` 디렉토리에 있는지 확인
- `agab_phase2_model.pth`가 존재하는지 확인

**"Out of memory" (메모리 부족)**
- CPU 사용: `predictor = AffinityPredictor(device='cpu')`
- 배치 크기 줄이기

**"Invalid amino acid" (잘못된 아미노산)**
- 서열이 유효한 아미노산만 포함하는지 확인 (ACDEFGHIKLMNPQRSTVWY)
- 특수 문자 및 갭 제거

**느린 첫 실행**
- 정상입니다! 첫 사용 시 ESM-2 모델 다운로드 (~140 MB)
- 후속 실행은 빠름 (모델이 캐시됨)

## 인용

연구에서 이 도구를 사용하는 경우 다음과 같이 인용해 주세요:

```bibtex
@software{abag_affinity,
  title={AbAg Binding Affinity Prediction},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/AbAg_binding_prediction}
}
```

## 라이선스

MIT 라이선스 - 자세한 내용은 LICENSE 파일을 참조하세요

## 기여

기여를 환영합니다! 이슈를 열거나 풀 리퀘스트를 제출해 주세요.

## 연락처

질문이나 지원이 필요한 경우 GitHub에서 이슈를 열어주세요.

---

**버전:** 1.0.0
**최종 업데이트:** 2025-10-31
**상태:** 프로덕션 준비 완료
