# 논문 적용 계획: Match, Compare, or Select?

**논문:** "Match, Compare, or Select? An Investigation of Large Language Models for Entity Matching"
**출처:** COLING 2025
**저자:** Tianshu Wang et al.

---

## 📚 논문 핵심 내용

### 3가지 Entity Matching 패러다임

1. **Matching (매칭)**: 두 entity를 직접 비교
   - "Entity A와 Entity B가 같은가?"
   - Binary decision (Yes/No)

2. **Comparison (비교)**: 여러 후보를 순위화
   - "Entity A는 Candidate 1, 2, 3 중 어디에 가장 가까운가?"
   - Ranking problem

3. **Selection (선택)**: 후보 풀에서 올바른 것 식별
   - "이 후보들 중 Entity A와 매칭되는 것은?"
   - Multi-choice selection

### 주요 발견

- ✅ **Selection과 Comparison이 Matching보다 우수**
- ✅ LLM은 ranking/filtering에 더 강함
- ✅ Binary decision보다 context-rich 비교에서 더 좋은 성능

---

## 🔍 현재 SNOMED CT 프로젝트 분석

### 현재 방식 (3rd Place)

```
1. Entity Recognition (span 추출)
   ↓
2. FAISS Retrieval (top-10 candidates)
   ↓
3. LLM Classification (top-10 중 하나 선택)
   "Which number (0-9) represents the correct code?"
```

**현재는 이미 "Selection" 패러다임!** ✅

하지만:
- ❌ 단순히 번호(0-9)만 선택
- ❌ 후보 간 비교 없음
- ❌ 명시적인 ranking 없음

---

## 💡 논문 기반 개선 방안

### **Approach 1: Explicit Comparison Paradigm** ⭐⭐⭐⭐⭐

현재 방식을 **Comparison 기반**으로 전환

#### Before (Selection):
```
Prompt:
"Here are 10 candidates (0-9). Select the number that matches."

Response:
"3"
```

#### After (Comparison):
```
Prompt:
"Term: chest pain
Context: patient presented with acute chest pain radiating to left arm

Compare these candidates:
1. Chest pain (finding) - SNOMED:29857009
2. Chest wall pain - SNOMED:78004001
3. Acute chest pain - SNOMED:12345678
...

For each candidate, explain:
- Semantic similarity to 'chest pain'
- Clinical relevance in this context
- Why it's better or worse than others

Then rank them from most to least relevant."

Response:
"Ranking:
1. Acute chest pain (best match - context mentions 'acute')
2. Chest pain (good match - direct synonym)
3. Chest wall pain (less likely - different location)
...
SELECTED: 3 (Acute chest pain)"
```

**기대 효과:**
- 더 나은 reasoning
- Context 활용 증가
- 정확도 +5-10%

---

### **Approach 2: Multi-Paradigm Ensemble** ⭐⭐⭐⭐⭐

3가지 패러다임을 **모두 사용**하고 앙상블

```
┌─────────────────────────────────────┐
│  Extracted Term: "chest pain"       │
│  Context: "acute onset..."          │
└─────────────────────────────────────┘
                ↓
        FAISS top-20 retrieval
                ↓
    ┌───────────┴───────────┬───────────────┐
    ↓                       ↓               ↓
┌─────────┐         ┌──────────┐     ┌────────────┐
│ Agent 1 │         │ Agent 2  │     │  Agent 3   │
│MATCHING │         │COMPARISON│     │ SELECTION  │
└─────────┘         └──────────┘     └────────────┘
    ↓                       ↓               ↓
Pairwise            Explicit            Top-1
comparison          ranking            selection
    ↓                       ↓               ↓
└───────────┬───────────────┴───────────────┘
            ↓
    ┌──────────────┐
    │ Voting/Merge │
    └──────────────┘
            ↓
    Final Prediction
```

#### Agent 1: MATCHING (Pairwise)

```python
def matching_agent(term, context, candidates):
    """각 후보와 직접 비교"""
    scores = []

    for candidate in candidates[:10]:
        prompt = f"""Does the term '{term}' in context:
"{context}"
MATCH with the SNOMED concept:
"{candidate.text}" (ID: {candidate.id})?

Consider:
- Exact semantic equivalence
- Clinical context
- Synonym relationships

Answer: YES/NO with confidence (0.0-1.0)
"""
        response = llm.generate(prompt)
        score = parse_yes_no_confidence(response)
        scores.append((candidate, score))

    return sorted(scores, key=lambda x: x[1], reverse=True)
```

#### Agent 2: COMPARISON (Ranking)

```python
def comparison_agent(term, context, candidates):
    """명시적으로 후보들을 비교하여 순위화"""
    prompt = f"""Term: {term}
Context: {context}

Candidates:
{format_candidates(candidates[:10])}

TASK: Compare ALL candidates and rank them.

For each pair, determine which is better:
- Candidate 1 vs 2: ?
- Candidate 1 vs 3: ?
- Candidate 2 vs 3: ?
...

Provide FINAL RANKING (1-10) with reasoning.
"""

    response = llm.generate(prompt)
    ranking = parse_ranking(response)

    return ranking
```

#### Agent 3: SELECTION (Current)

```python
def selection_agent(term, context, candidates):
    """현재 방식: 후보 중 선택"""
    prompt = f"""Term: {term}
Context: {context}

Candidates:
0: {candidate[0]}
1: {candidate[1]}
...

Select the number (0-9) or -1 if none match.
"""

    response = llm.generate(prompt)
    selected = parse_selection(response)

    return selected
```

#### Ensemble

```python
def ensemble_decision(matching_result, comparison_result, selection_result):
    """3가지 결과를 종합"""

    # Voting
    votes = defaultdict(float)
    votes[matching_result[0]] += 0.4  # 40% weight
    votes[comparison_result[0]] += 0.35  # 35% weight
    votes[selection_result] += 0.25  # 25% weight

    # Confidence weighting
    final = max(votes.items(), key=lambda x: x[1])

    return final
```

**기대 효과:**
- 더 robust한 예측
- 각 패러다임의 장점 결합
- 정확도 +10-18%

---

### **Approach 3: Hierarchical Selection** ⭐⭐⭐⭐

단계적으로 후보를 줄여가며 선택

```
FAISS top-20
    ↓
┌─────────────────────────┐
│ Stage 1: Coarse Filter  │  ← COMPARISON
│ (20 → 5)                │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ Stage 2: Fine Ranking   │  ← MATCHING
│ (5 → 2)                 │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ Stage 3: Final Selection│  ← SELECTION
│ (2 → 1)                 │
└─────────────────────────┘
```

#### Stage 1: Coarse Filtering (Comparison)

```python
def coarse_filter(term, context, top_20):
    """빠른 비교로 top-5 선정"""
    prompt = f"""Term: {term}
Context: {context}

20 Candidates (brief):
{format_brief(top_20)}

Quickly compare and select TOP-5 most relevant.
Consider only semantic similarity.
"""

    top_5 = llm.generate(prompt)
    return parse_top_k(top_5, k=5)
```

#### Stage 2: Fine Ranking (Matching)

```python
def fine_ranking(term, context, top_5):
    """상세 pairwise 비교"""
    # 모든 쌍 비교
    comparison_matrix = []

    for i in range(len(top_5)):
        for j in range(i+1, len(top_5)):
            prompt = f"""Which is better match for '{term}' in context?

A) {top_5[i].full_description}
B) {top_5[j].full_description}

Analyze clinical nuances and choose A or B.
"""
            winner = llm.generate(prompt)
            comparison_matrix.append((i, j, winner))

    # Bradley-Terry model for ranking
    scores = compute_scores_from_comparisons(comparison_matrix)

    return sorted(top_5, key=lambda x: scores[x.id], reverse=True)[:2]
```

#### Stage 3: Final Selection

```python
def final_selection(term, context, top_2):
    """최종 선택"""
    prompt = f"""Term: {term}
Full context: {context}

Final two candidates:
1) {top_2[0].full_info}
2) {top_2[1].full_info}

Which one is the CORRECT SNOMED CT code for this term?
Provide detailed reasoning.
"""

    final = llm.generate(prompt)
    return parse_final_choice(final)
```

**기대 효과:**
- 계산 효율성 (단계적 필터링)
- 높은 정확도 (상세 비교)
- 정확도 +12-20%

---

### **Approach 4: Multi-Agent Debate + 논문 패러다임** ⭐⭐⭐⭐⭐

기존 multi-agent 제안과 논문 아이디어 결합

```
┌─────────────────────────────────────┐
│  Term + Context + FAISS top-20      │
└─────────────────────────────────────┘
                ↓
    ┌───────────┴───────────────┐
    ↓                           ↓
┌──────────────┐        ┌──────────────┐
│ Matching     │        │ Comparison   │
│ Specialist   │        │ Specialist   │
│ Committee    │        │ Committee    │
└──────────────┘        └──────────────┘
    ↓                           ↓
  3 agents                   3 agents
  pairwise                   ranking
    ↓                           ↓
    └───────────┬───────────────┘
                ↓
        ┌──────────────┐
        │ Debate Agent │
        └──────────────┘
                ↓
        ┌──────────────┐
        │   Consensus  │
        └──────────────┘
```

---

## 🎯 추천 구현 순서

### **Phase 1: Quick Win (1-2주)** ⭐ 시작하기

**Approach 1 구현: Explicit Comparison**

현재 classification prompt를 comparison 기반으로 변경

**파일:** `src/snomedctentitylinking.py`

**수정 위치:** `improve_assign_condition()` 함수

```python
# Before
prompt = f"""Select the number (0-9) that matches:
Term: {term}
Candidates: {faiss_results}
"""

# After
prompt = f"""Compare and rank these candidates for '{term}':

Context: {context}
Section: {section}

Candidates:
{format_detailed_candidates(faiss_results)}

TASK:
1. For each candidate, analyze:
   - Semantic similarity
   - Clinical relevance
   - Contextual appropriateness

2. Compare candidates pairwise:
   - Which pairs are similar?
   - Which are clearly inferior?

3. Provide final ranking (1-10)

4. Select the BEST match (or -1 if none appropriate)

Format:
RANKING: [ordered list]
SELECTED: [number]
REASONING: [brief explanation]
"""
```

**예상 개선:** +5-8% classification accuracy

---

### **Phase 2: Multi-Paradigm Ensemble (2-3주)**

**Approach 2 구현**

```python
# multi_paradigm_classifier.py

class MultiParadigmClassifier:
    def __init__(self, llm_model):
        self.matching_agent = MatchingAgent(llm_model)
        self.comparison_agent = ComparisonAgent(llm_model)
        self.selection_agent = SelectionAgent(llm_model)

    def classify(self, term, context, section, faiss_results):
        # 3가지 방식으로 모두 분류
        match_result = self.matching_agent.classify(...)
        comp_result = self.comparison_agent.classify(...)
        sel_result = self.selection_agent.classify(...)

        # Ensemble
        final = self.ensemble([match_result, comp_result, sel_result])

        return final
```

**예상 개선:** +10-15% classification accuracy

---

### **Phase 3: Hierarchical + Multi-Agent (3-4주)**

**Approach 3 + Approach 4 결합**

가장 강력한 조합

**예상 개선:** +15-25% overall IoU

---

## 📊 예상 성능 비교

| 방법 | Classification Acc | IoU | 구현 난이도 | 추론 비용 |
|------|-------------------|-----|------------|----------|
| **Current (Selection only)** | 0.65 | 0.378 | - | 1.0x |
| **+ Explicit Comparison** | **0.70-0.73** | **0.41-0.43** | 쉬움 | 1.2x |
| **+ Multi-Paradigm** | **0.75-0.78** | **0.45-0.48** | 중간 | 2.5x |
| **+ Hierarchical** | **0.78-0.82** | **0.48-0.52** | 어려움 | 2.0x |
| **All Combined** | **0.82-0.86** | **0.52-0.58** | 어려움 | 3.0x |

---

## 💻 즉시 적용 가능한 코드

### Comparison-Based Prompt (Phase 1)

```python
def create_comparison_prompt(term, context, section, faiss_results):
    """논문의 Comparison 패러다임 적용"""

    candidates_text = []
    for i, (concept_id, score, text) in enumerate(faiss_results[:10]):
        candidates_text.append(
            f"{i}. {text} (ID: {concept_id}, similarity: {score:.3f})"
        )

    prompt = f"""<|im_start|>system
You are an expert medical coder specializing in SNOMED CT.<|im_end|>
<|im_start|>user
TASK: Compare and rank SNOMED CT candidates for entity matching.

TERM: "{term}"
CONTEXT: "{context}"
SECTION: "{section}"

CANDIDATES:
{chr(10).join(candidates_text)}

INSTRUCTIONS:
1. COMPARE each candidate:
   a) Semantic similarity to the term
   b) Clinical appropriateness in context
   c) SNOMED CT concept specificity

2. RANK candidates (best to worst):
   - Explain why top candidates are better
   - Note why bottom candidates are worse

3. SELECT the best match:
   - Choose 0-9 for the best candidate
   - Choose -1 if NO candidate is appropriate

FORMAT:
COMPARISON: [detailed pairwise analysis]
RANKING: [5, 2, 0, 1, ...]
SELECTED: [number]
CONFIDENCE: [0.0-1.0]
REASONING: [why this is the best match]
<|im_end|>
<|im_start|>assistant
"""

    return prompt


# Integration with existing code
def improve_assign_condition_with_comparison(
    model_path,
    model_path_cache,
    df_notes,
    faiss_db,
    df_dict,
    is_submission
):
    """기존 함수를 comparison 방식으로 교체"""
    import src.model.vLLM as LLM

    llm = LLM.instantiate(model_path, model_path_cache)

    for index_o, row in df_notes.iterrows():
        prompts = []

        for inst in row["result_chunks_inst"]:
            term_context = extract_context(row["text"], int(inst[1]), int(inst[2]))

            faiss_results = [
                (inst[6].split(",")[i],
                 inst[7].split(",")[i],
                 inst[8].split(",")[i])
                for i in range(min(10, len(inst[6].split(","))))
            ]

            # NEW: Comparison-based prompt
            prompt = create_comparison_prompt(
                term=inst[5],
                context=term_context,
                section=inst[9],
                faiss_results=faiss_results
            )

            prompts.append(prompt)

        # Batch inference
        prompts_inf = LLM.inference(llm, prompts)

        # Parse responses (새로운 파서 필요)
        for index_i, response in enumerate(prompts_inf):
            parsed = parse_comparison_response(response)

            if parsed['selected'] >= 0 and parsed['confidence'] >= 0.6:
                concept_id = inst[6].split(",")[parsed['selected']]
                df_notes.at[index_o, "result_chunks_inst"][index_i][3] = concept_id

        if not is_submission:
            print(f"Document {index_o} classified with comparison method")


def parse_comparison_response(response):
    """Comparison 형식의 응답 파싱"""
    import re

    # SELECTED 추출
    selected_match = re.search(r'SELECTED:\s*(-?\d+)', response)
    selected = int(selected_match.group(1)) if selected_match else -1

    # CONFIDENCE 추출
    conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response)
    confidence = float(conf_match.group(1)) if conf_match else 0.5

    # RANKING 추출 (optional, 나중에 사용)
    ranking_match = re.search(r'RANKING:\s*\[(.*?)\]', response)
    ranking = []
    if ranking_match:
        ranking = [int(x.strip()) for x in ranking_match.group(1).split(',')]

    return {
        'selected': selected,
        'confidence': confidence,
        'ranking': ranking,
        'raw_response': response
    }
```

---

## 🔬 실험 설계

### A/B Testing

| Group | Method | Expected IoU |
|-------|--------|--------------|
| **A (Baseline)** | Current Selection | 0.378 |
| **B (Comparison)** | Explicit Comparison | 0.41-0.43 |
| **C (Multi-Paradigm)** | 3-agent ensemble | 0.45-0.48 |
| **D (Hierarchical)** | Staged filtering | 0.48-0.52 |

### 평가 지표

1. **Classification Accuracy @1**
2. **Classification Accuracy @5** (top-5에 정답 포함)
3. **IoU Score** (최종)
4. **Confidence Calibration** (predicted confidence vs actual accuracy)

---

## 🎯 최종 추천

### **단기 (2주)**: Approach 1
- ✅ 구현 쉬움
- ✅ 즉시 효과
- ✅ 기존 코드 최소 수정
- ✅ 비용 증가 적음 (+20%)

### **중기 (1-2개월)**: Approach 2 + Approach 1
- ✅ 최고 정확도
- ✅ Robust prediction
- ⚠️ 추론 비용 증가 (2.5x)

### **장기 (연구용)**: Approach 4
- ✅ SOTA 달성 가능
- ✅ 논문 출판 가능
- ⚠️ 복잡도 높음
- ⚠️ 비용 높음 (3x)

---

## 📚 논문 인용

```bibtex
@inproceedings{wang-etal-2025-match,
    title = "Match, Compare, or Select? An Investigation of Large Language Models for Entity Matching",
    author = "Wang, Tianshu and Chen, Xiaoyang and Lin, Hongyou and Chen, Xuanang and Han, Xianpei and Sun, Le and Wang, Hao and Zeng, Zhenyu",
    booktitle = "Proceedings of COLING 2025",
    year = "2025",
    url = "https://aclanthology.org/2025.coling-main.8"
}
```

---

## ✅ Next Steps

1. **즉시**: Comparison prompt 구현 및 테스트
2. **1주 후**: 성능 비교 분석
3. **2주 후**: Multi-paradigm ensemble 구현
4. **1개월 후**: 논문 작성 시작

궁금한 점이나 구현 도움이 필요하면 말씀해주세요!
