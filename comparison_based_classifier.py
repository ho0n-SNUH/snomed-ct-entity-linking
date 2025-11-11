"""
Comparison-Based Classification for SNOMED CT Entity Linking

Based on: "Match, Compare, or Select?" (COLING 2025)
Paper: https://aclanthology.org/2025.coling-main.8.pdf

Key Insight: LLMs perform better with COMPARISON paradigm than simple SELECTION
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import re


@dataclass
class ComparisonResult:
    """Classification 결과"""
    selected_index: int  # -1 if no match
    selected_concept_id: str
    confidence: float
    ranking: List[int]  # Ordered indices
    reasoning: str
    raw_response: str


class ComparisonBasedClassifier:
    """
    논문의 Comparison 패러다임을 적용한 classifier

    기존 Selection 방식:
        "이 후보들 중 번호를 선택하세요"

    새로운 Comparison 방식:
        "후보들을 비교하고 순위를 매긴 후 최선을 선택하세요"
    """

    def __init__(
        self,
        llm_model,
        use_detailed_comparison: bool = True,
        use_pairwise_analysis: bool = True,
        min_confidence: float = 0.6
    ):
        """
        Args:
            llm_model: LLM inference model
            use_detailed_comparison: 상세한 비교 분석 사용 여부
            use_pairwise_analysis: Pairwise 비교 포함 여부
            min_confidence: 최소 confidence threshold
        """
        self.llm_model = llm_model
        self.use_detailed_comparison = use_detailed_comparison
        self.use_pairwise_analysis = use_pairwise_analysis
        self.min_confidence = min_confidence

    def create_comparison_prompt(
        self,
        term: str,
        context: str,
        section: str,
        faiss_results: List[Tuple[str, float, str]],
        use_qwen_format: bool = True
    ) -> str:
        """
        Comparison 기반 프롬프트 생성

        Args:
            term: 분류할 의료 용어
            context: 주변 문맥
            section: 문서 섹션
            faiss_results: [(concept_id, similarity, text), ...]
            use_qwen_format: Qwen2.5 chat format 사용

        Returns:
            Comparison prompt
        """

        # Candidates 포맷팅
        candidates_text = []
        for i, (concept_id, score, text) in enumerate(faiss_results):
            candidates_text.append(
                f"  {i}. {text}\n"
                f"     ID: {concept_id}\n"
                f"     Similarity: {score:.3f}"
            )

        candidates_str = "\n".join(candidates_text)

        # Pairwise analysis 지시사항
        pairwise_instruction = ""
        if self.use_pairwise_analysis and len(faiss_results) <= 10:
            pairwise_instruction = """
3. PAIRWISE COMPARISON (important):
   Compare key candidates against each other:
   - Which pairs are semantically similar?
   - Which candidates clearly dominate others?
   - Are there any subtle clinical differences?
"""

        # System message
        system_msg = """You are an expert medical coder specializing in SNOMED CT.
Your task is to match clinical terms to the correct SNOMED CT concepts through systematic comparison."""

        # User message
        user_msg = f"""TASK: Compare and rank SNOMED CT candidates, then select the best match.

TERM TO MATCH: "{term}"

CLINICAL CONTEXT:
{context}

DOCUMENT SECTION: {section}

CANDIDATE CONCEPTS:
{candidates_str}

INSTRUCTIONS:

1. INDIVIDUAL ANALYSIS:
   For each candidate, evaluate:
   a) Semantic similarity to the term "{term}"
   b) Clinical appropriateness in this specific context
   c) SNOMED CT concept specificity and accuracy
   d) Common usage patterns in clinical documentation

2. COMPARATIVE ANALYSIS:
   Compare candidates systematically:
   - Which candidates are most/least similar to the term?
   - Which are more/less appropriate for this clinical context?
   - Which are more/less specific SNOMED CT concepts?
{pairwise_instruction}

4. RANKING:
   Provide a complete ranking from best to worst (0-{len(faiss_results)-1})

5. SELECTION:
   Choose the BEST match:
   - Select 0-{len(faiss_results)-1} for the best candidate
   - Select -1 if NO candidate is clinically appropriate

6. CONFIDENCE:
   Rate your confidence (0.0-1.0):
   - 1.0 = Certain match
   - 0.7-0.9 = High confidence
   - 0.5-0.7 = Moderate confidence
   - < 0.5 = Low confidence (consider -1)

RESPONSE FORMAT:
COMPARISON: [Your detailed comparison analysis]
RANKING: [comma-separated indices, e.g., 5,2,0,1,3,4,6,7,8,9]
SELECTED: [single number 0-{len(faiss_results)-1}, or -1]
CONFIDENCE: [0.0-1.0]
REASONING: [Why this is the best match, or why no match]"""

        # Format for Qwen2.5
        if use_qwen_format:
            prompt = f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
{user_msg}<|im_end|>
<|im_start|>assistant
"""
        else:
            # Mistral format
            prompt = f"[INST] {system_msg}\n\n{user_msg} [/INST]"

        return prompt

    def parse_response(
        self,
        response: str,
        faiss_results: List[Tuple[str, float, str]]
    ) -> ComparisonResult:
        """
        LLM 응답 파싱

        Args:
            response: LLM의 raw response
            faiss_results: Original candidates

        Returns:
            ComparisonResult
        """

        # SELECTED 추출
        selected_match = re.search(
            r'SELECTED:\s*(-?\d+)',
            response,
            re.IGNORECASE
        )
        selected_idx = int(selected_match.group(1)) if selected_match else -1

        # CONFIDENCE 추출
        conf_match = re.search(
            r'CONFIDENCE:\s*([\d.]+)',
            response,
            re.IGNORECASE
        )
        confidence = float(conf_match.group(1)) if conf_match else 0.5

        # RANKING 추출
        ranking_match = re.search(
            r'RANKING:\s*\[?([\d,\s]+)\]?',
            response,
            re.IGNORECASE
        )
        ranking = []
        if ranking_match:
            try:
                ranking = [
                    int(x.strip())
                    for x in ranking_match.group(1).split(',')
                    if x.strip()
                ]
            except ValueError:
                ranking = []

        # REASONING 추출
        reasoning_match = re.search(
            r'REASONING:\s*(.+?)(?=\n\n|\n[A-Z]+:|\Z)',
            response,
            re.IGNORECASE | re.DOTALL
        )
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        # Concept ID 가져오기
        selected_concept_id = ""
        if 0 <= selected_idx < len(faiss_results):
            selected_concept_id = faiss_results[selected_idx][0]

        return ComparisonResult(
            selected_index=selected_idx,
            selected_concept_id=selected_concept_id,
            confidence=confidence,
            ranking=ranking,
            reasoning=reasoning,
            raw_response=response
        )

    def classify(
        self,
        term: str,
        context: str,
        section: str,
        faiss_results: List[Tuple[str, float, str]]
    ) -> ComparisonResult:
        """
        단일 term 분류

        Args:
            term: 분류할 의료 용어
            context: 주변 문맥
            section: 문서 섹션
            faiss_results: FAISS 검색 결과

        Returns:
            ComparisonResult
        """

        # Prompt 생성
        prompt = self.create_comparison_prompt(
            term, context, section, faiss_results
        )

        # LLM inference
        response = self.llm_model.generate([prompt])[0]

        # 응답 파싱
        result = self.parse_response(response, faiss_results)

        # Confidence filtering
        if result.confidence < self.min_confidence:
            result.selected_index = -1
            result.selected_concept_id = ""

        return result

    def classify_batch(
        self,
        terms: List[str],
        contexts: List[str],
        sections: List[str],
        faiss_results_batch: List[List[Tuple[str, float, str]]]
    ) -> List[ComparisonResult]:
        """
        Batch classification

        Args:
            terms: 용어 리스트
            contexts: 문맥 리스트
            sections: 섹션 리스트
            faiss_results_batch: FAISS 결과 리스트

        Returns:
            ComparisonResult 리스트
        """

        # Batch prompts 생성
        prompts = [
            self.create_comparison_prompt(term, ctx, sec, faiss)
            for term, ctx, sec, faiss in zip(
                terms, contexts, sections, faiss_results_batch
            )
        ]

        # Batch inference
        responses = self.llm_model.generate(prompts)

        # 파싱
        results = [
            self.parse_response(resp, faiss)
            for resp, faiss in zip(responses, faiss_results_batch)
        ]

        # Confidence filtering
        for result in results:
            if result.confidence < self.min_confidence:
                result.selected_index = -1
                result.selected_concept_id = ""

        return results


# ============================================================================
# Integration with existing 3rd Place code
# ============================================================================

def integrate_comparison_classifier(
    model_path: str,
    model_path_cache: str,
    df_notes,
    is_submission: bool = True
):
    """
    기존 improve_assign_condition()을 Comparison 방식으로 교체

    Drop-in replacement for:
        src/snomedctentitylinking.py::improve_assign_condition()

    Args:
        model_path: Classification model path
        model_path_cache: Model cache directory
        df_notes: Notes DataFrame with FAISS results
        is_submission: Submission mode flag
    """
    import src.model.vLLM as LLM

    # LLM 로드
    llm = LLM.instantiate(model_path, model_path_cache)

    # Comparison classifier 초기화
    from model_migration_guide import QwenModelWrapper

    # Qwen 사용 시
    if "Qwen" in model_path or "qwen" in model_path:
        llm_wrapper = QwenModelWrapper(model_path, use_vllm=True)
    else:
        # 기존 Mistral 등
        llm_wrapper = llm

    classifier = ComparisonBasedClassifier(
        llm_model=llm_wrapper,
        use_detailed_comparison=True,
        use_pairwise_analysis=True,
        min_confidence=0.6
    )

    # 각 note 처리
    for index_o, row in df_notes.iterrows():
        terms = []
        contexts = []
        sections = []
        faiss_results_batch = []
        original_insts = []

        # Batch 데이터 준비
        for inst in row["result_chunks_inst"]:
            # Context 추출
            term_context = extract_context(
                row["text"],
                int(inst[1]),
                int(inst[2])
            )

            # FAISS 결과 포맷팅
            faiss_results = []
            concept_ids = inst[6].split(",")
            scores = inst[7].split(",")
            texts = inst[8].split(",")

            for i in range(min(10, len(concept_ids))):
                faiss_results.append((
                    concept_ids[i],
                    float(scores[i]) if scores[i] else 0.0,
                    texts[i]
                ))

            terms.append(inst[5])  # term text
            contexts.append(term_context)
            sections.append(inst[9])  # section
            faiss_results_batch.append(faiss_results)
            original_insts.append(inst)

        # Batch classification
        results = classifier.classify_batch(
            terms, contexts, sections, faiss_results_batch
        )

        # 결과 업데이트
        list_to_remove = []
        for inst, result in zip(original_insts, results):
            if result.selected_index >= 0:
                # Concept ID 업데이트
                index_i = row["result_chunks_inst"].index(inst)
                df_notes.at[index_o, "result_chunks_inst"][index_i][3] = \
                    result.selected_concept_id
            else:
                # Low confidence or no match → 제거
                list_to_remove.append(inst)

        # 낮은 confidence 항목 제거
        df_notes.at[index_o, "result_chunks_inst"] = [
            inst
            for inst in df_notes.at[index_o, "result_chunks_inst"]
            if inst not in list_to_remove
        ]

        if not is_submission:
            print(f"Document {index_o}: Classified with comparison method")
            print(f"  - Kept: {len(results) - len(list_to_remove)}")
            print(f"  - Removed (low conf): {len(list_to_remove)}")


def extract_context(
    text: str,
    start: int,
    end: int,
    num_words_before: int = 5,
    num_words_after: int = 5
) -> str:
    """Extract context around term (from original code)"""
    words = text.split()
    words_before = text[:start].split()
    words_after = text[end:].split()
    start_word_index = len(words_before)
    end_word_index = len(words) - len(words_after) - 1

    start_context_index = max(0, start_word_index - num_words_before)
    end_context_index = min(len(words), end_word_index + num_words_after + 1)

    words_before_target = words[start_context_index:start_word_index]
    target_words = words[start_word_index : end_word_index + 1]
    words_after_target = words[end_word_index + 1 : end_context_index]

    return " ".join(words_before_target + target_words + words_after_target)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("""
Comparison-Based Classification for SNOMED CT Entity Linking

Based on COLING 2025 paper:
"Match, Compare, or Select? An Investigation of Large Language Models for Entity Matching"

Usage:

1. Drop-in replacement in main.py:
   ----------------------------------------
   # Before:
   improve_assign_condition(model_path, ...)

   # After:
   from comparison_based_classifier import integrate_comparison_classifier
   integrate_comparison_classifier(model_path, ...)


2. Standalone usage:
   ----------------------------------------
   from comparison_based_classifier import ComparisonBasedClassifier
   from model_migration_guide import QwenModelWrapper

   llm = QwenModelWrapper("Qwen/Qwen2.5-7B-Instruct")
   classifier = ComparisonBasedClassifier(llm)

   result = classifier.classify(
       term="chest pain",
       context="patient presented with acute chest pain",
       section="Assessment",
       faiss_results=[
           ("29857009", 0.89, "Chest pain (finding)"),
           ("12345678", 0.85, "Acute chest pain"),
           ...
       ]
   )

   print(f"Selected: {result.selected_concept_id}")
   print(f"Confidence: {result.confidence}")
   print(f"Reasoning: {result.reasoning}")


Expected Improvements:
   - Classification accuracy: +5-10%
   - Overall IoU: +8-12%
   - Better handling of ambiguous cases
   - More interpretable decisions (explicit reasoning)

""")
