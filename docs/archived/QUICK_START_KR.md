# 빠른 시작 가이드 - AbAg 결합 친화도 예측

5분 안에 항체-항원 결합 친화도 예측을 시작하세요!

## 설치

```bash
# 패키지 디렉토리로 이동
cd /mnt/c/Users/401-24/Desktop/AbAg_binding_prediction

# 의존성 설치
pip install -r requirements.txt

# 패키지 설치 (개발 모드)
pip install -e .
```

## 첫 번째 예측

### 단계 1: 예측기 가져오기

```python
from abag_affinity import AffinityPredictor

# 초기화 (자동으로 모델 로드)
predictor = AffinityPredictor()
```

**첫 실행 참고**: ESM-2 모델 다운로드 (~140 MB), 이후 실행은 빠릅니다!

### 단계 2: 예측 수행

```python
# 항체 및 항원 서열
antibody_heavy = "EVQLQQSGPGLVKPSQTLSLTCAISG..."
antigen = "KVFGRCELAAAMKRHGLDNYRGYSLGN..."

# 결합 친화도 예측
result = predictor.predict(
    antibody_heavy=antibody_heavy,
    antigen=antigen
)

# 결과 확인
print(f"pKd: {result['pKd']:.2f}")
print(f"Kd: {result['Kd_nM']:.1f} nM")
print(f"카테고리: {result['category']}")
```

### 단계 3: 결과 해석

| pKd | Kd | 해석 |
|-----|-----|------|
| > 10 | < 1 nM | 예외적인 결합체 (치료용 품질) |
| 9-10 | 1-10 nM | 매우 강한 결합체 |
| 7.5-9 | 10-100 nM | 강한 결합체 (연구용 품질) |
| 6-7.5 | 0.1-10 μM | 중간 결합체 |
| < 6 | > 10 μM | 약하거나 비결합체 |

## 테스트 실행

설치 확인:

```bash
python tests/test_installation.py
```

예상 출력:
```
🎉 ALL TESTS PASSED!
Your installation is working correctly!
```

## 예제 실행

전체 예제 보기:

```bash
python examples/basic_usage.py
```

다음을 시연합니다:
1. 단일 예측
2. Heavy chain만 사용하는 항체
3. 배치 처리
4. 변이체 비교

## 일반적인 사용 사례

### 항체 라이브러리 스크리닝

```python
# 항체 라이브러리 로드
library = [
    {'id': 'Ab001', 'heavy': 'EVQ...', 'antigen': target},
    {'id': 'Ab002', 'heavy': 'QVQ...', 'antigen': target},
    # ... 더 많은 후보
]

# 배치 예측
results = predictor.predict_batch(library)

# 친화도로 정렬
best = sorted(results, key=lambda x: x['pKd'], reverse=True)

# 상위 10개 후보
for r in best[:10]:
    print(f"{r['id']}: pKd={r['pKd']:.2f}, Kd={r['Kd_nM']:.1f} nM")
```

### 돌연변이체 비교

```python
# 원래 항체
original = predictor.predict(heavy=wild_type, antigen=target)

# 돌연변이 테스트
mutant = predictor.predict(heavy=mutated, antigen=target)

# 비교
improvement = mutant['pKd'] - original['pKd']
print(f"ΔpKd: {improvement:+.2f}")
```

## 문제 해결

**"Model not found" (모델을 찾을 수 없음)**
- `models/agab_phase2_model.pth`가 존재하는지 확인
- 패키지 루트 디렉토리에서 실행

**"Out of memory" (메모리 부족)**
- CPU 사용: `predictor = AffinityPredictor(device='cpu')`
- 배치 크기 줄이기

**"Invalid amino acid" (잘못된 아미노산)**
- 서열이 다음만 포함하는지 확인: ACDEFGHIKLMNPQRSTVWY
- 갭 및 특수 문자 제거

## 다음 단계

1. **전체 문서 읽기**: [README_KR.md](README_KR.md)
2. **예제 시도**: `python examples/basic_usage.py`
3. **데이터 처리**: 배치 처리 예제 적용
4. **API 탐색**: `help(AffinityPredictor)`로 docstring 확인

## 성능 참고사항

- **정확도**: Spearman ρ = 0.85, Pearson r = 0.95
- **속도**: 예측당 ~1-2초 (GPU), ~5-10초 (CPU)
- **배치 처리**: 개별 예측보다 훨씬 빠름
- **첫 실행**: ESM-2 다운로드로 인해 느림, 이후 실행은 캐시됨

## 질문이 있으신가요?

- 전체 README 확인: [README_KR.md](README_KR.md)
- 테스트 실행: `python tests/test_installation.py`
- 예제 시도: `python examples/basic_usage.py`

---

**이제 결합 친화도를 예측할 준비가 되었습니다! 🚀**
