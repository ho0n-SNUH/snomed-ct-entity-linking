"""
Multi-Agent Classification Example for SNOMED CT Entity Linking

이 예제는 현재 3rd Place 솔루션의 classification 단계를
multi-agent 방식으로 개선하는 방법을 보여줍니다.

Phase 1 구현: Debate-Based Classification
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import Counter
import numpy as np


@dataclass
class ClassificationResult:
    """각 agent의 분류 결과"""
    agent_name: str
    concept_id: str
    confidence: float
    reasoning: str


class MultiAgentClassifier:
    """
    Multi-Agent Classification System

    여러 agent가 다른 전략으로 동일한 term을 분류하고,
    debate와 voting을 통해 최종 결정
    """

    def __init__(self, llm_model, faiss_db, df_dict):
        self.llm_model = llm_model
        self.faiss_db = faiss_db
        self.df_dict = df_dict

        # 다양한 전략을 가진 agent들
        self.agents = {
            "semantic_agent": self._create_semantic_agent(),
            "context_agent": self._create_context_agent(),
            "frequency_agent": self._create_frequency_agent(),
            "conservative_agent": self._create_conservative_agent(),
        }

    def classify_with_debate(
        self,
        term: str,
        context: str,
        section: str,
        faiss_results: List[Tuple[str, float, str]]
    ) -> Tuple[str, float]:
        """
        Multi-agent debate를 통한 분류

        Args:
            term: 분류할 의료 용어
            context: 주변 문맥
            section: 문서 섹션
            faiss_results: FAISS 검색 결과 [(concept_id, score, text), ...]

        Returns:
            (최종 concept_id, confidence)
        """

        # Round 1: 각 agent가 독립적으로 분류
        initial_results = self._round1_initial_classification(
            term, context, section, faiss_results
        )

        # Round 2: Agent들이 서로의 결과를 보고 재분류
        refined_results = self._round2_cross_review(
            term, context, section, faiss_results, initial_results
        )

        # Round 3: Consensus building
        final_concept_id, confidence = self._round3_consensus(
            refined_results, initial_results
        )

        return final_concept_id, confidence

    def _round1_initial_classification(
        self, term, context, section, faiss_results
    ) -> List[ClassificationResult]:
        """Round 1: 각 agent가 독립적으로 분류"""
        results = []

        for agent_name, agent_strategy in self.agents.items():
            result = agent_strategy(term, context, section, faiss_results)
            results.append(result)

        return results

    def _round2_cross_review(
        self, term, context, section, faiss_results, initial_results
    ) -> List[ClassificationResult]:
        """Round 2: Agent들이 다른 agent의 결과를 보고 재평가"""
        refined_results = []

        for agent_name, agent_strategy in self.agents.items():
            # 다른 agent들의 결과를 함께 제공
            other_results = [r for r in initial_results if r.agent_name != agent_name]

            # 재평가 프롬프트
            refinement_prompt = self._create_refinement_prompt(
                term, context, section, faiss_results, other_results
            )

            # Agent가 다른 의견들을 고려해서 재분류
            result = self._refine_classification(
                agent_name, agent_strategy, refinement_prompt,
                term, context, section, faiss_results
            )
            refined_results.append(result)

        return refined_results

    def _round3_consensus(
        self, refined_results, initial_results
    ) -> Tuple[str, float]:
        """Round 3: Voting과 confidence scoring으로 최종 결정"""

        # Strategy 1: Majority Voting
        concept_ids = [r.concept_id for r in refined_results]
        vote_counts = Counter(concept_ids)

        # Strategy 2: Weighted by Confidence
        concept_scores = {}
        for result in refined_results:
            if result.concept_id not in concept_scores:
                concept_scores[result.concept_id] = []
            concept_scores[result.concept_id].append(result.confidence)

        # 각 concept의 평균 confidence
        avg_confidences = {
            cid: np.mean(scores) for cid, scores in concept_scores.items()
        }

        # Vote count와 confidence를 결합
        final_scores = {}
        for concept_id, count in vote_counts.items():
            # 투표 수와 confidence의 가중 평균
            final_scores[concept_id] = (
                count * 0.6 + avg_confidences[concept_id] * len(refined_results) * 0.4
            )

        # 최고 점수의 concept 선택
        best_concept = max(final_scores, key=final_scores.get)

        # Confidence 계산: agreement + avg confidence
        agreement = vote_counts[best_concept] / len(refined_results)
        avg_conf = avg_confidences[best_concept]
        final_confidence = (agreement * 0.5 + avg_conf * 0.5)

        return best_concept, final_confidence

    # ========== Agent Strategy Creators ==========

    def _create_semantic_agent(self):
        """Semantic similarity에 집중하는 agent"""
        def strategy(term, context, section, faiss_results):
            prompt = f"""You are a medical coding specialist focusing on SEMANTIC SIMILARITY.

Term to classify: {term}
Context: {context}
Section: {section}

Top FAISS matches:
{self._format_faiss_results(faiss_results[:10])}

Focus on:
1. Which concept has the most similar MEANING to the term?
2. Consider synonyms and medical terminology variations
3. Prioritize semantic match over frequency

Respond with:
CONCEPT_ID: <id>
CONFIDENCE: <0.0-1.0>
REASONING: <brief explanation>
"""

            response = self._call_llm(prompt)
            return self._parse_agent_response(response, "semantic_agent")

        return strategy

    def _create_context_agent(self):
        """Context와 section을 중시하는 agent"""
        def strategy(term, context, section, faiss_results):
            prompt = f"""You are a medical coding specialist focusing on CLINICAL CONTEXT.

Term to classify: {term}
Context: {context}
Section: {section}

Top FAISS matches:
{self._format_faiss_results(faiss_results[:10])}

Focus on:
1. Which concept makes MOST SENSE in this clinical context?
2. Consider the section (e.g., "Assessment" vs "Medications")
3. Consider surrounding medical terms and patient status
4. Avoid codes that are semantically similar but contextually wrong

Respond with:
CONCEPT_ID: <id>
CONFIDENCE: <0.0-1.0>
REASONING: <brief explanation>
"""

            response = self._call_llm(prompt)
            return self._parse_agent_response(response, "context_agent")

        return strategy

    def _create_frequency_agent(self):
        """FAISS score와 통계를 중시하는 agent"""
        def strategy(term, context, section, faiss_results):
            prompt = f"""You are a medical coding specialist focusing on STATISTICAL PATTERNS.

Term to classify: {term}
Context: {context}

Top FAISS matches with scores:
{self._format_faiss_results_with_scores(faiss_results[:10])}

Focus on:
1. Which concept has the BEST FAISS similarity score?
2. Is there a clear winner or are scores close?
3. Consider the score distribution pattern
4. Higher scores generally indicate better matches

Respond with:
CONCEPT_ID: <id>
CONFIDENCE: <0.0-1.0>
REASONING: <brief explanation>
"""

            response = self._call_llm(prompt)
            return self._parse_agent_response(response, "frequency_agent")

        return strategy

    def _create_conservative_agent(self):
        """Conservative하게 확실한 경우만 선택"""
        def strategy(term, context, section, faiss_results):
            prompt = f"""You are a CONSERVATIVE medical coding specialist.

Term to classify: {term}
Context: {context}
Section: {section}

Top FAISS matches:
{self._format_faiss_results(faiss_results[:10])}

Focus on:
1. Only choose if you are HIGHLY CONFIDENT
2. If ambiguous, prefer -1 (no match) over incorrect match
3. Consider medical accuracy over recall
4. Verify the match is clinically appropriate

Respond with:
CONCEPT_ID: <id or -1>
CONFIDENCE: <0.0-1.0>
REASONING: <brief explanation>
"""

            response = self._call_llm(prompt)
            return self._parse_agent_response(response, "conservative_agent")

        return strategy

    # ========== Helper Methods ==========

    def _create_refinement_prompt(
        self, term, context, section, faiss_results, other_results
    ) -> str:
        """다른 agent들의 의견을 포함한 refinement prompt"""
        others_summary = "\n".join([
            f"- {r.agent_name}: chose {r.concept_id} (confidence: {r.confidence:.2f})"
            f"\n  Reasoning: {r.reasoning}"
            for r in other_results
        ])

        prompt = f"""Term: {term}
Context: {context}
Section: {section}

Other agents' opinions:
{others_summary}

FAISS results:
{self._format_faiss_results(faiss_results[:10])}

Consider:
1. Do you agree with other agents?
2. Did they catch something you missed?
3. Are there valid alternative interpretations?
4. Update your classification if needed
"""
        return prompt

    def _refine_classification(
        self, agent_name, agent_strategy, refinement_prompt,
        term, context, section, faiss_results
    ) -> ClassificationResult:
        """Agent가 다른 의견을 고려해서 재분류"""

        # 원래 전략 실행
        initial = agent_strategy(term, context, section, faiss_results)

        # Refinement prompt 추가
        full_prompt = f"""{refinement_prompt}

Your initial choice was: {initial.concept_id}
Do you want to change it? Explain why or why not.

Respond with:
CONCEPT_ID: <id>
CONFIDENCE: <0.0-1.0>
REASONING: <brief explanation>
"""

        response = self._call_llm(full_prompt)
        refined = self._parse_agent_response(response, agent_name)

        return refined

    def _format_faiss_results(self, results: List[Tuple]) -> str:
        """FAISS 결과를 읽기 좋게 포맷팅"""
        formatted = []
        for i, (concept_id, score, text) in enumerate(results):
            formatted.append(f"{i}: {concept_id} - {text}")
        return "\n".join(formatted)

    def _format_faiss_results_with_scores(self, results: List[Tuple]) -> str:
        """FAISS 결과를 score 포함해서 포맷팅"""
        formatted = []
        for i, (concept_id, score, text) in enumerate(results):
            formatted.append(f"{i}: {concept_id} (score: {score:.4f}) - {text}")
        return "\n".join(formatted)

    def _call_llm(self, prompt: str) -> str:
        """LLM 호출 (실제 구현 필요)"""
        # 실제로는 vLLM이나 다른 inference engine 사용
        # return self.llm_model.generate(prompt)
        pass

    def _parse_agent_response(
        self, response: str, agent_name: str
    ) -> ClassificationResult:
        """Agent 응답을 파싱"""
        # 실제 구현에서는 regex나 structured output parsing
        lines = response.strip().split('\n')
        concept_id = None
        confidence = 0.0
        reasoning = ""

        for line in lines:
            if line.startswith("CONCEPT_ID:"):
                concept_id = line.split(":", 1)[1].strip()
            elif line.startswith("CONFIDENCE:"):
                confidence = float(line.split(":", 1)[1].strip())
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        return ClassificationResult(
            agent_name=agent_name,
            concept_id=concept_id,
            confidence=confidence,
            reasoning=reasoning
        )


# ========== Integration with existing code ==========

def improve_assign_condition_multiagent(
    model_path,
    model_path_cache,
    df_notes,
    faiss_db,
    df_dict,
    is_submission
):
    """
    기존 improve_assign_condition 함수를 multi-agent로 대체

    This replaces the single LLM classification with multi-agent debate
    """
    import src.model.vLLM as LLM

    llm = LLM.instantiate(model_path, model_path_cache)

    # Multi-agent classifier 초기화
    classifier = MultiAgentClassifier(llm, faiss_db, df_dict)

    for index_o, row in df_notes.iterrows():
        results_with_high_confidence = []
        results_needing_debate = []

        # 먼저 간단한 케이스 필터링 (최적화)
        for inst in row["result_chunks_inst"]:
            term_context = extract_context(row["text"], int(inst[1]), int(inst[2]))
            top_concepts_id = inst[6].split(",")
            top_concepts_score = [float(s) for s in inst[7].split(",")]
            top_concepts_text = inst[8].split(",")

            faiss_results = list(zip(
                top_concepts_id[:20],
                top_concepts_score[:20],
                top_concepts_text[:20]
            ))

            # 만약 FAISS top-1이 매우 높은 score면 debate 생략
            if top_concepts_score[0] > 0.95:
                results_with_high_confidence.append((inst, top_concepts_id[0]))
            else:
                results_needing_debate.append((inst, faiss_results, term_context))

        # Debate가 필요한 것들만 multi-agent로 처리
        for inst, faiss_results, term_context in results_needing_debate:
            concept_id, confidence = classifier.classify_with_debate(
                term=inst[5],
                context=term_context,
                section=inst[9],
                faiss_results=faiss_results
            )

            # confidence가 낮으면 제거 (더 보수적)
            if concept_id != "-1" and confidence >= 0.6:
                results_with_high_confidence.append((inst, concept_id))

        # 결과 업데이트
        list_to_remove = []
        for inst, concept_id in results_with_high_confidence:
            index_i = row["result_chunks_inst"].index(inst)
            df_notes.at[index_o, "result_chunks_inst"][index_i][3] = concept_id

        # Confidence 낮은 것들은 제거
        all_processed_insts = [inst for inst, _ in results_with_high_confidence]
        df_notes.at[index_o, "result_chunks_inst"] = [
            inst for inst in df_notes.at[index_o, "result_chunks_inst"]
            if inst in all_processed_insts
        ]

        if not is_submission:
            print(f"Document {index_o}: {len(results_with_high_confidence)} terms classified")


def extract_context(text: str, start: int, end: int,
                    num_words_before: int = 5,
                    num_words_after: int = 5) -> str:
    """현재 코드에서 가져온 함수"""
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


# ========== Usage Example ==========

if __name__ == "__main__":
    """
    사용 예시:

    기존 코드의 improve_assign_condition을
    improve_assign_condition_multiagent로 교체
    """

    # main.py에서:
    # improve_assign_condition(...) 대신
    # improve_assign_condition_multiagent(...) 사용

    print("Multi-Agent Classification System")
    print("=" * 50)
    print("\n4개의 specialized agents:")
    print("1. Semantic Agent - 의미적 유사성 중심")
    print("2. Context Agent - 임상적 맥락 중심")
    print("3. Frequency Agent - 통계적 패턴 중심")
    print("4. Conservative Agent - 보수적 판단")
    print("\n3 Round Debate:")
    print("Round 1: 독립적 분류")
    print("Round 2: 상호 검토 및 재분류")
    print("Round 3: Consensus building")
    print("\n기대 효과:")
    print("- Classification accuracy: +7-12%")
    print("- False positive 감소")
    print("- Confidence scoring 개선")
