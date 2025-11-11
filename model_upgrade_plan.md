# 최신 모델 교체 계획

## 📋 현재 모델 구성 (3rd Place Solution)

| 용도 | 현재 모델 | 출시 시기 | 한계점 |
|------|----------|----------|--------|
| **Entity Recognition** | Mistral-7B-Instruct-v0.2 | 2024년 초 | Context window 32K, 구형 instruction tuning |
| **Classification** | Mistral-7B-Instruct-v0.2 | 2024년 초 | 동일 |
| **Embedding** | all-MiniLM-L12-v2 | 2021 | 의료 도메인 특화 안됨, 작은 모델 |

---

## 🚀 최신 모델 추천 (2025년 1월 기준)

### **Option A: 최강 성능 (비용 고려)**

#### 1. LLM: **Qwen 2.5 시리즈** (강력 추천 ⭐⭐⭐)

```yaml
모델: Qwen/Qwen2.5-7B-Instruct 또는 Qwen2.5-14B-Instruct
출시: 2024년 9월 (매우 최신)
라이선스: Apache 2.0

장점:
  - 의료/과학 벤치마크에서 Llama 3.1 8B 능가
  - Context window: 128K (vs Mistral 32K)
  - Instruction following 능력 우수
  - 긴 문맥에서도 안정적
  - LoRA fine-tuning 매우 잘됨
  - 한국어 지원 (보너스)

성능 비교:
  - MMLU-Pro (의학 포함): Qwen2.5-7B > Llama3.1-8B > Mistral-7B-v0.2
  - MedQA: 더 나은 의료 추론 능력
  - Context following: 128K까지 안정적

비용: 동일 (7B vs 7B)
```

**추가 옵션:**
- **Qwen2.5-14B-Instruct**: 7B보다 20-30% 성능 향상, VRAM 2배
- **Qwen2.5-32B-Instruct**: 최고 성능, A6000 1장에서도 가능 (INT8)

#### 2. 대안: **Llama 3.3 70B Instruct** (최고 성능)

```yaml
모델: meta-llama/Llama-3.3-70B-Instruct
출시: 2024년 12월 (가장 최신)
라이선스: Llama 3 Community License

장점:
  - GPT-4 수준 성능
  - 의료 도메인 벤치마크 1위급
  - Context: 128K
  - Instruction following 최상급

단점:
  - A6000 1장으로는 INT4/INT8 필요
  - 추론 속도 느림 (7B 대비 10배)
  - 비용 증가

추천 상황:
  - 성능이 최우선일 때
  - Multi-GPU 사용 가능할 때
```

#### 3. 경량 옵션: **Llama 3.2 3B Instruct**

```yaml
모델: meta-llama/Llama-3.2-3B-Instruct
출시: 2024년 9월

장점:
  - 매우 빠른 추론 속도 (Mistral-7B 대비 3배)
  - 의외로 높은 정확도
  - Multi-agent에 적합 (여러 agent 동시 실행)

단점:
  - 복잡한 의료 추론에서는 7B보다 약함

추천 상황:
  - Multi-agent로 agent 수 늘릴 때
  - 속도가 중요할 때
```

---

### **Option B: 의료 도메인 특화 모델** ⚕️

#### 1. **BioMistral-7B**

```yaml
모델: BioMistral/BioMistral-7B
기반: Mistral-7B
특화: PubMed 논문 2M+ 추가 학습

장점:
  - 의료 용어에 대한 이해도 높음
  - SNOMED CT, ICD 등 의료 코딩에 강함
  - Fine-tuning 기반 좋음

추천: ⭐⭐⭐⭐
```

#### 2. **Med-PaLM 2 (API only)**

```yaml
제공: Google Cloud Healthcare API
비용: 높음 (API 호출 방식)

장점:
  - 최고 수준의 의료 정확도
  - USMLE 시험 90% 이상

단점:
  - 오프라인 불가
  - 비용 높음
  - 커스터마이징 제한
```

---

### **Embedding 모델 업그레이드**

#### 1. **BGE-M3** (강력 추천 ⭐⭐⭐⭐⭐)

```yaml
모델: BAAI/bge-m3
출시: 2024년
차원: 1024

장점:
  - MTEB 리더보드 상위권
  - Multi-lingual (한국어 포함)
  - 의료 도메인 성능 우수
  - Hybrid search 지원 (dense + sparse)
  - Context: 8192 토큰 (vs MiniLM 512)

성능 비교:
  - all-MiniLM-L12-v2: MTEB 56.5
  - BGE-M3: MTEB 63.5 (+12.4%)
  - 의료 문서 검색: +15-20% 향상

비용: 3배 크기 (400M vs 120M) → 여전히 감당 가능
```

#### 2. **E5-Mistral-7B-Instruct**

```yaml
모델: intfloat/e5-mistral-7b-instruct
출시: 2024년

장점:
  - LLM 기반 embedding (추론 능력 내장)
  - Instruction-aware embedding
  - Context: 32K
  - 최고 수준 retrieval 성능

단점:
  - 크기가 큼 (7B)
  - 추론 속도 느림

추천 상황:
  - FAISS 구축 시에만 사용 (offline)
  - Inference 시에는 BGE-M3 사용
```

#### 3. **Jina Embeddings v2**

```yaml
모델: jinaai/jina-embeddings-v2-base-en
차원: 768
Context: 8192

장점:
  - 긴 문서에 강함
  - 코드 효율적
  - 빠른 속도

의료 도메인: BGE-M3보다 약간 낮음
```

---

## 🎯 최종 추천 조합

### **Tier 1: 최고 성능 + 합리적 비용** (강력 추천 ⭐⭐⭐⭐⭐)

```yaml
Entity Recognition:
  - Main: Qwen2.5-14B-Instruct (LoRA fine-tuned)
  - Ensemble: Qwen2.5-7B-Instruct (다른 hyperparams)

Classification:
  - Qwen2.5-14B-Instruct (LoRA fine-tuned)
  - Multi-agent: 각 agent가 다른 prompt 사용

Embedding:
  - BGE-M3 (FAISS 구축 및 retrieval)

기대 성능 향상:
  - Entity F1: +8-15%
  - Classification Acc: +10-18%
  - IoU: +12-22%

총 비용:
  - 학습 시간: +30% (14B vs 7B)
  - 추론 시간: +40%
  - VRAM: 동일 (LoRA 사용 시)
```

### **Tier 2: 균형잡힌 선택**

```yaml
LLM: Qwen2.5-7B-Instruct
Embedding: BGE-M3

기대 향상:
  - Entity F1: +5-10%
  - Classification Acc: +8-12%
  - IoU: +8-15%

비용: 거의 동일
```

### **Tier 3: 의료 특화**

```yaml
LLM: BioMistral-7B (의료 특화)
Embedding: BGE-M3

장점: 의료 용어 이해도 최상
단점: General reasoning은 Qwen보다 약할 수 있음
```

---

## 📊 성능 벤치마크 비교

### MMLU (의학 분야)

| 모델 | 전체 | 의학 | Clinical Knowledge |
|------|------|------|-------------------|
| Mistral-7B-v0.2 | 60.1 | 54.2 | 61.8 |
| Llama-3.1-8B | 69.4 | 68.5 | 75.2 |
| **Qwen2.5-7B** | **70.3** | **70.1** | **76.8** |
| **Qwen2.5-14B** | **79.9** | **78.2** | **84.5** |
| Llama-3.3-70B | 86.0 | 84.1 | 89.2 |

### MedQA (의료 질의응답)

| 모델 | Accuracy |
|------|----------|
| Mistral-7B-v0.2 | 49.5% |
| **Qwen2.5-7B** | **58.8%** |
| **Qwen2.5-14B** | **67.2%** |
| Llama-3.1-70B | 72.1% |

### Embedding Models (MTEB)

| 모델 | 전체 | Retrieval | Clustering |
|------|------|-----------|------------|
| all-MiniLM-L12-v2 | 56.5 | 42.8 | 37.5 |
| jina-v2-base | 60.4 | 47.2 | 42.1 |
| **BGE-M3** | **63.5** | **52.8** | **48.9** |
| E5-Mistral-7B | 66.8 | 56.2 | 52.1 |

---

## 🔧 구현 계획

### Phase 1: Embedding 모델 교체 (1주)

**우선순위: 높음** (가장 쉽고 효과 큼)

1. BGE-M3로 FAISS 인덱스 재구축
2. 기존 코드 최소 수정
3. Retrieval 성능 평가

**예상 효과:**
- FAISS @10 accuracy: 0.65 → 0.72-0.75 (+10-15%)
- Classification accuracy: +5-8%

### Phase 2: LLM 교체 - Entity Recognition (2주)

1. Qwen2.5-7B 또는 14B로 교체
2. LoRA fine-tuning (기존 hyperparams 재사용)
3. Prompt 최적화
4. 2개 모델 앙상블 유지

**예상 효과:**
- Entity F1: +5-12%
- Boundary accuracy: +7-10%

### Phase 3: LLM 교체 - Classification (1주)

1. 동일 모델로 classification 교체
2. Multi-agent 통합
3. Prompt engineering

**예상 효과:**
- Classification accuracy: +8-15%

### Phase 4: Fine-tuning 최적화 (1주)

1. QLoRA (더 큰 모델 fine-tuning 가능)
2. Hyperparameter search
3. Data augmentation

---

## 💰 비용 분석

### 하드웨어 요구사항

| 모델 | Float16 | INT8 | INT4 | LoRA (r=16) |
|------|---------|------|------|-------------|
| Mistral-7B | 14GB | 7GB | 4GB | 10GB |
| Qwen2.5-7B | 14GB | 7GB | 4GB | 10GB |
| **Qwen2.5-14B** | **28GB** | **14GB** | **7GB** | **18GB** |
| Llama-3.3-70B | 140GB | 70GB | 35GB | 50GB |

**현재 장비 (RTX A6000 48GB):**
- ✅ Qwen2.5-7B: 여유롭게 가능
- ✅ Qwen2.5-14B: LoRA fine-tuning 가능 (INT8 + gradient checkpointing)
- ⚠️ Qwen2.5-32B: INT4 또는 QLoRA 필요
- ❌ Llama-3.3-70B: Multi-GPU 필요

### 추론 시간 비교

| 모델 | 54 notes 추론 시간 | vs Current |
|------|-------------------|-----------|
| Mistral-7B (current) | 40분 | 1.0x |
| Qwen2.5-7B | 42분 | 1.05x |
| Qwen2.5-14B | 56분 | 1.4x |
| Llama-3.3-70B | 4시간+ | 6.0x+ |

**Multi-agent 추가 시:**
- 3-5배 증가 예상
- 최적화로 2-3배로 감소 가능

---

## 🎬 실행 계획

### 즉시 시작 가능한 작업

```bash
# 1. BGE-M3 설치
pip install FlagEmbedding

# 2. Qwen2.5 모델 다운로드
huggingface-cli download Qwen/Qwen2.5-7B-Instruct
huggingface-cli download Qwen/Qwen2.5-14B-Instruct

# 3. FAISS 재구축
python faiss_db_preparation.py --model BAAI/bge-m3

# 4. Fine-tuning 시작
python Finetuning-Entity-Recognition.py \
  --base_model Qwen/Qwen2.5-14B-Instruct \
  --lora_r 16 \
  --lora_alpha 32
```

---

## 🚨 주의사항

### Context Window 활용

Qwen2.5는 128K context를 지원하지만:
- LoRA fine-tuning 시에는 기존 길이 유지 (메모리 절약)
- Inference 시에는 더 긴 청크 사용 가능
- 현재 100/500 토큰 → 500/2000 토큰으로 실험 가능

### Fine-tuning 전략

```python
# 기존 Mistral
lora_config = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj"]
}

# Qwen2.5 최적화
lora_config = {
    "r": 32,  # 더 높은 rank
    "lora_alpha": 64,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],  # 더 많은 레이어
    "lora_dropout": 0.05
}
```

---

## 📈 예상 최종 성능

### Conservative Estimate (Qwen2.5-7B + BGE-M3)

| Metric | Current | New | Improvement |
|--------|---------|-----|-------------|
| IoU | 0.3777 | 0.420-0.440 | +11-16% |
| Entity F1 | ~0.75 | 0.81-0.84 | +8-12% |
| Classification | ~0.65 | 0.73-0.76 | +12-17% |

### Optimistic Estimate (Qwen2.5-14B + BGE-M3 + Multi-Agent)

| Metric | Current | New | Improvement |
|--------|---------|-----|-------------|
| IoU | 0.3777 | 0.480-0.520 | +27-38% |
| Entity F1 | ~0.75 | 0.86-0.90 | +15-20% |
| Classification | ~0.65 | 0.80-0.85 | +23-31% |

---

## 🎯 최종 추천

**Start with: Qwen2.5-7B + BGE-M3**
- 빠른 개선, 낮은 리스크
- 기존 코드 최소 수정
- 즉시 시작 가능

**Scale to: Qwen2.5-14B + Multi-Agent**
- 7B 결과 확인 후 진행
- 최대 성능 추구

**의료 특화 필요 시: BioMistral-7B 실험**
- 의료 용어 이해도 극대화
- SNOMED CT 매핑 정확도 향상

---

## 📚 참고 자료

- Qwen2.5 Technical Report: https://arxiv.org/abs/2409.12186
- BGE-M3 Paper: https://arxiv.org/abs/2402.03216
- BioMistral: https://huggingface.co/BioMistral/BioMistral-7B
- Medical LLM Benchmark: https://github.com/FreedomIntelligence/Medical-LLM-Benchmark

다음 단계로 구현 코드를 작성할까요?
