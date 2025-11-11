# LLM Multi-Agent 아키텍처 제안서

## 현재 시스템 분석

### 3rd Place 솔루션 구조
1. **Entity Recognition**: Mistral-7B 2개 모델 (100 토큰 / 500 토큰 청크)
2. **Retrieval**: FAISS vector search (top-10)
3. **Classification**: Mistral-7B 분류 모델

### 한계점
- 단방향 파이프라인: 에러가 누적됨
- Agent 간 상호작용 없음: 집단 지성 활용 못함
- 단일 관점: 하나의 reasoning path만 고려

---

## 🎯 Multi-Agent 개선 방안

### **방안 1: Collaborative Extraction + Debate-Based Classification**

#### 아키텍처
```
Clinical Note Input
        ↓
┌─────────────────────────────────────┐
│  Phase 1: Multi-Agent Extraction    │
│  ├─ Aggressive Extractor (recall↑)  │
│  ├─ Precise Extractor (precision↑)  │
│  └─ Context Extractor (contextual)  │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│  Phase 2: Cross-Validation          │
│  - Agent들이 서로의 결과 검토         │
│  - 불일치 항목에 대해 토론            │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│  Phase 3: Consensus Building        │
│  - Voting mechanism                 │
│  - Confidence scoring               │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│  Phase 4: Classification Committee  │
│  - 3-5 agents가 각자 분류            │
│  - FAISS top-20 각자 다르게 해석     │
│  - Final vote                       │
└─────────────────────────────────────┘
```

#### 기대 효과
- **Recall 향상**: 여러 agent가 다른 전략으로 추출
- **Precision 향상**: Cross-validation으로 false positive 제거
- **Classification 정확도**: 여러 관점에서 FAISS 결과 해석

#### 구현 복잡도
- 중간 (LangGraph 사용 시)
- 기존 코드 40-50% 재활용 가능

---

### **방안 2: Hierarchical Domain Specialist System**

#### 아키텍처
```
Input Note
    ↓
[Coordinator Agent] ← 전체 orchestration
    ↓
[Section Parser] ← 문서 구조 파악
    ↓
[Domain Router] ← 각 section을 전문 agent에 할당
    ↓
┌────────┬─────────┬─────────┬─────────┐
│ Cardio │ Pulmo   │ Neuro   │ Renal   │ Domain Specialists
│ Agent  │ Agent   │ Agent   │ Agent   │
└────────┴─────────┴─────────┴─────────┘
    ↓
[Consistency Checker] ← 중복/모순 해결
    ↓
[Ensemble Classifier] ← 최종 SNOMED CT 매핑
```

#### Domain-Specific Prompts 예시
```
Cardiovascular Agent:
"You are a cardiologist. Focus on:
- Heart conditions (e.g., CHF, CAD, arrhythmia)
- Blood pressure related terms
- Cardiac procedures
..."

Pulmonary Agent:
"You are a pulmonologist. Focus on:
- Respiratory conditions
- Breathing patterns
- Lung function
..."
```

#### 기대 효과
- **도메인 정확도**: 각 분야별 전문성 극대화
- **Long-tail 처리**: 희귀 질환도 전문 agent가 처리
- **Scalability**: 새 도메인 agent 추가 용이

---

### **방안 3: Self-Reflection & Iterative Refinement**

#### 아키텍처
```
Round 1: Initial Extraction
    [Agent A] [Agent B] [Agent C]
        ↓
Round 2: Peer Review
    [Critic Agent] reviews each output
        ↓
Round 3: Self-Refinement
    Each agent revises based on feedback
        ↓
Round 4: Meta-Critic
    [Meta-Critic] identifies remaining issues
        ↓
Round 5: Final Consensus
    [Consensus Agent] makes final decision
```

#### Prompt Flow 예시
```
Round 1 Prompt:
"Extract medical entities from this text."

Round 2 Prompt (to Critic):
"Review these extractions. For each entity:
1. Is it a valid medical term?
2. Are boundaries correct?
3. Is it clinically significant?
Provide feedback."

Round 3 Prompt (back to extractor):
"Here's your original extraction and peer feedback.
Revise your extraction considering:
- Feedback points
- Boundary adjustments
- Missing entities"
```

#### 기대 효과
- **Error Correction**: 반복적 개선으로 에러 수정
- **Boundary Precision**: span 경계 정확도 향상
- **False Positive 감소**: 여러 검토 단계

---

### **방안 4: RAG-Enhanced Multi-Agent Classification**

#### 현재 문제점
- FAISS만 사용 → 단일 정보원
- Top-10만 고려 → 제한적 context

#### 개선 방안
```
For each extracted entity:
    ↓
┌─────────────────────────────────────┐
│  Multi-Source Retrieval             │
│  ├─ FAISS (semantic similarity)     │
│  ├─ SNOMED CT Graph (relationships) │
│  ├─ Training Annotations (examples) │
│  └─ Medical KB (definitions)        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Reasoning Agent Pool               │
│  - Agent 1: Focus on FAISS results  │
│  - Agent 2: Focus on graph context  │
│  - Agent 3: Focus on examples       │
│  - Agent 4: Focus on definitions    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Ensemble Classification            │
│  - Weighted voting                  │
│  - Confidence thresholding          │
└─────────────────────────────────────┘
```

#### 추가 구현 요소
1. **SNOMED CT Graph Navigation**
   - Parent/child relationships
   - Synonyms and related concepts

2. **Case-Based Reasoning**
   - Similar cases from training data
   - Context matching

3. **Knowledge Base Integration**
   - Medical definitions
   - Clinical guidelines

---

## 🚀 추천 구현 순서

### Phase 1: Quick Win (1-2주)
**Debate-Based Classification** 구현
- 현재 classification 단계만 multi-agent로 변경
- Entity recognition은 기존 유지
- 기대 효과: Classification accuracy 5-10% 향상

### Phase 2: Core Improvement (2-3주)
**Collaborative Extraction** 추가
- 3개 extraction agent 구현
- Cross-validation 로직 추가
- 기대 효과: F1 score 5-15% 향상

### Phase 3: Advanced (1-2주)
**Self-Reflection Loop** 추가
- Critic agent 구현
- Iterative refinement
- 기대 효과: Boundary accuracy 향상

### Phase 4: Optimization (1주)
- Domain specialists (optional)
- Multi-source RAG (optional)

---

## 📊 예상 성능 향상

| Metric | Current (3rd Place) | Multi-Agent (Conservative) | Multi-Agent (Optimistic) |
|--------|---------------------|----------------------------|--------------------------|
| IoU | 0.3777 | 0.42-0.45 | 0.47-0.50 |
| Entity F1 | ~0.75 (추정) | 0.80-0.82 | 0.85-0.88 |
| Classification Acc | ~0.65 (FAISS@20) | 0.72-0.75 | 0.78-0.82 |

---

## 💻 기술 스택 제안

### Option A: LangGraph + LangChain
```python
from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Multi-agent workflow 정의
workflow = StateGraph()
workflow.add_node("extractor_1", aggressive_extract)
workflow.add_node("extractor_2", precise_extract)
workflow.add_node("critic", review_extractions)
workflow.add_node("consensus", build_consensus)
...
```

**장점**:
- Workflow 시각화
- 상태 관리 용이
- 디버깅 편리

### Option B: AutoGen (Microsoft)
```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat

extractor_1 = AssistantAgent("AggressiveExtractor", ...)
extractor_2 = AssistantAgent("PreciseExtractor", ...)
critic = AssistantAgent("Critic", ...)

group_chat = GroupChat(
    agents=[extractor_1, extractor_2, critic],
    messages=[],
    max_round=3
)
```

**장점**:
- Multi-agent 대화 자동 관리
- Group chat 기능 내장

### Option C: Custom Implementation
- 완전한 제어
- 기존 코드 재활용 극대화
- 추론 비용 최적화 가능

---

## 💰 비용 및 리소스 고려사항

### 추론 비용 증가
- 기존 대비 3-5배 API 호출 증가 예상
- 최적화 방법:
  1. 불확실한 경우만 multi-agent 활성화
  2. 작은 모델로 pre-filtering
  3. 캐싱 적극 활용

### GPU 메모리
- 여러 agent 병렬 실행 시 VRAM 증가
- 해결: Sequential execution or smaller batch

---

## 🔬 실험 설계

### A/B Testing
1. Baseline: 현재 3rd place 솔루션
2. Variant A: Classification만 multi-agent
3. Variant B: Full multi-agent pipeline
4. Variant C: Domain specialists

### 평가 지표
- Primary: IoU score
- Secondary: Entity F1, Classification Accuracy
- Tertiary: Inference time, Cost per note

---

## 📖 참고 문헌

1. **Multi-Agent Debate**: "Improving Factuality and Reasoning in Language Models through Multiagent Debate" (Du et al., 2023)
2. **Self-Consistency**: "Self-Consistency Improves Chain of Thought Reasoning in Language Models" (Wang et al., 2022)
3. **AutoGen**: Microsoft AutoGen framework
4. **LangGraph**: LangChain multi-agent workflows

---

## ✅ Next Steps

1. **현재 코드 분석** (완료)
2. **Multi-agent 프레임워크 선택** (LangGraph 추천)
3. **Phase 1 구현**: Debate-based classification
4. **평가 및 반복**
5. **Phase 2-4 순차 진행**

## 질문 사항

1. 추론 비용 증가 허용 범위는?
2. 선호하는 프레임워크 (LangGraph vs AutoGen vs Custom)?
3. 실험용 GPU 리소스 현황?
4. Timeline 제약사항?
