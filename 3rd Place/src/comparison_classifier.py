"""
Comparison-Based Classification for SNOMED CT
Based on COLING 2025 paper: "Match, Compare, or Select?"

This module provides a drop-in replacement for the original classification method.
"""

import re
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class ComparisonResult:
    """Classification result with detailed information"""
    selected_index: int  # -1 if no match
    selected_concept_id: str
    confidence: float
    ranking: List[int]
    reasoning: str
    raw_response: str


def create_comparison_prompt_qwen(
    term: str,
    context: str,
    section: str,
    faiss_list: List[Tuple[str, str, str]]
) -> str:
    """
    Create comparison-based prompt for Qwen2.5

    Args:
        term: The term to classify (with context markers)
        context: Surrounding context
        section: Document section
        faiss_list: [(concept_id, score, text), ...]

    Returns:
        Formatted prompt for Qwen2.5
    """
    # Format candidates
    candidates_text = []
    for i, (concept_id, score, text) in enumerate(faiss_list):
        candidates_text.append(
            f"  {i}. {text}\n"
            f"     ID: {concept_id}\n"
            f"     Similarity: {score}"
        )

    candidates_str = "\n".join(candidates_text)

    prompt = f"""<|im_start|>system
You are an expert medical coder specializing in SNOMED CT.
Your task is to match clinical terms to correct SNOMED CT concepts through systematic comparison.<|im_end|>
<|im_start|>user
TASK: Compare and rank SNOMED CT candidates, then select the best match.

TERM TO MATCH: "{term}"

CLINICAL CONTEXT:
{context}

DOCUMENT SECTION: {section}

CANDIDATE CONCEPTS:
{candidates_str}

INSTRUCTIONS:

1. INDIVIDUAL ANALYSIS:
   For each candidate, evaluate:
   - Semantic similarity to the term
   - Clinical appropriateness in this specific context
   - SNOMED CT concept specificity

2. COMPARATIVE ANALYSIS:
   - Which candidates are most/least similar?
   - Which are more/less appropriate for this context?
   - Any subtle clinical differences?

3. RANKING:
   Provide complete ranking from best to worst (0-{len(faiss_list)-1})

4. SELECTION:
   - Select 0-{len(faiss_list)-1} for the best candidate
   - Select -1 if NO candidate is appropriate

5. CONFIDENCE:
   Rate confidence (0.0-1.0):
   - 1.0 = Certain match
   - 0.7-0.9 = High confidence
   - 0.5-0.7 = Moderate
   - <0.5 = Low (consider -1)

RESPONSE FORMAT:
COMPARISON: [Your detailed analysis]
RANKING: [comma-separated indices, e.g., 5,2,0,1,3]
SELECTED: [number or -1]
CONFIDENCE: [0.0-1.0]
REASONING: [Brief explanation]<|im_end|>
<|im_start|>assistant
"""

    return prompt


def create_comparison_prompt_mistral(
    term: str,
    context: str,
    section: str,
    faiss_list: List[Tuple[str, str, str]]
) -> str:
    """
    Create comparison-based prompt for Mistral format

    Compatible with original Mistral-7B models
    """
    # Format candidates
    candidates_text = []
    for i, (concept_id, score, text) in enumerate(faiss_list):
        candidates_text.append(f"{i}: {concept_id} - {text} (score: {score})")

    candidates_str = "\n".join(candidates_text)

    prompt = f"""[INST]You are an expert medical coder. Your task is to match clinical terms to SNOMED CT concepts through comparison.

TERM: {term}
CONTEXT: {context}
SECTION: {section}

CANDIDATES:
{candidates_str}

Compare these candidates systematically:
1. Analyze each candidate's semantic similarity and clinical relevance
2. Compare candidates against each other
3. Rank them from best to worst
4. Select the best match (0-{len(faiss_list)-1}) or -1 if none appropriate
5. Provide your confidence (0.0-1.0)

Respond with:
SELECTED: [number]
CONFIDENCE: [0.0-1.0]
REASONING: [brief explanation]
[/INST]"""

    return prompt


def parse_comparison_response(
    response: str,
    faiss_list: List[Tuple[str, str, str]]
) -> ComparisonResult:
    """
    Parse LLM response from comparison prompt

    Args:
        response: Raw LLM output
        faiss_list: Original candidates

    Returns:
        ComparisonResult with parsed information
    """
    # Extract SELECTED
    selected_match = re.search(
        r'SELECTED:\s*(-?\d+)',
        response,
        re.IGNORECASE
    )
    selected_idx = int(selected_match.group(1)) if selected_match else -1

    # Extract CONFIDENCE
    conf_match = re.search(
        r'CONFIDENCE:\s*([\d.]+)',
        response,
        re.IGNORECASE
    )
    confidence = float(conf_match.group(1)) if conf_match else 0.5

    # Extract RANKING
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
        except (ValueError, AttributeError):
            ranking = []

    # Extract REASONING
    reasoning_match = re.search(
        r'REASONING:\s*(.+?)(?=\n\n|\n[A-Z]+:|\Z)',
        response,
        re.IGNORECASE | re.DOTALL
    )
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    # Get concept ID
    selected_concept_id = ""
    if 0 <= selected_idx < len(faiss_list):
        selected_concept_id = faiss_list[selected_idx][0]

    return ComparisonResult(
        selected_index=selected_idx,
        selected_concept_id=selected_concept_id,
        confidence=confidence,
        ranking=ranking,
        reasoning=reasoning,
        raw_response=response
    )


def improve_assign_condition_comparison(
    model_path: str,
    model_path_cache: str,
    df_notes,
    classification_template: str,  # Not used, kept for compatibility
    is_submission: bool,
    min_confidence: float = 0.6,
    use_qwen_format: bool = None
):
    """
    Improved classification using comparison paradigm

    Drop-in replacement for improve_assign_condition() with better performance.

    Args:
        model_path: Path to classification model
        model_path_cache: Model cache directory
        df_notes: DataFrame with extracted entities and FAISS results
        classification_template: Original template (ignored, kept for compatibility)
        is_submission: Whether this is for submission
        min_confidence: Minimum confidence threshold (default: 0.6)
        use_qwen_format: Use Qwen chat format (auto-detect if None)

    Returns:
        None (modifies df_notes in place)
    """
    import src.model.vLLM as LLM

    # Load model
    llm = LLM.instantiate(model_path, model_path_cache)

    # Auto-detect model type
    if use_qwen_format is None:
        use_qwen_format = "qwen" in model_path.lower() or "Qwen" in model_path

    if not is_submission:
        print(f"Using comparison-based classification (COLING 2025 method)")
        print(f"Model type: {'Qwen' if use_qwen_format else 'Mistral'}")
        print(f"Confidence threshold: {min_confidence}")

    # Process each document
    for index_o, row in df_notes.iterrows():
        prompts = []
        faiss_lists = []
        original_insts = []

        # Prepare batch of prompts
        for inst in row["result_chunks_inst"]:
            # Extract context
            term_context = extract_context(row["text"], int(inst[1]), int(inst[2]))

            # Get FAISS results
            top_concepts_id = inst[6].split(",")
            top_concepts_score = inst[7].split(",")
            top_concepts_text = inst[8].split(",")

            faiss_list = list(
                zip(
                    top_concepts_id[:10],
                    top_concepts_score[:10],
                    top_concepts_text[:10]
                )
            )

            # Create comparison prompt
            if use_qwen_format:
                prompt = create_comparison_prompt_qwen(
                    term=inst[5],
                    context=term_context,
                    section=inst[9],
                    faiss_list=faiss_list
                )
            else:
                prompt = create_comparison_prompt_mistral(
                    term=inst[5],
                    context=term_context,
                    section=inst[9],
                    faiss_list=faiss_list
                )

            prompts.append(prompt)
            faiss_lists.append(faiss_list)
            original_insts.append(inst)

        # Batch inference
        prompts_inf = LLM.inference(llm, prompts)

        # Parse results and update
        list_to_remove = []
        high_conf_count = 0
        low_conf_count = 0

        for index_i, (response, inst, faiss_list) in enumerate(
            zip(prompts_inf, original_insts, faiss_lists)
        ):
            # Parse response
            result = parse_comparison_response(response, faiss_list)

            # Apply confidence threshold
            if result.confidence >= min_confidence and result.selected_index >= 0:
                # Update concept ID
                inst_index = row["result_chunks_inst"].index(inst)
                df_notes.at[index_o, "result_chunks_inst"][inst_index][3] = \
                    result.selected_concept_id
                high_conf_count += 1
            else:
                # Low confidence or no match → remove
                list_to_remove.append(inst)
                low_conf_count += 1

                if not is_submission and result.selected_index >= 0:
                    print(f"  Low confidence ({result.confidence:.2f}): {inst[4]}")

        # Remove low-confidence predictions
        df_notes.at[index_o, "result_chunks_inst"] = [
            inst
            for inst in df_notes.at[index_o, "result_chunks_inst"]
            if inst not in list_to_remove
        ]

        if not is_submission:
            print(f"Document {index_o}:")
            print(f"  High confidence: {high_conf_count}")
            print(f"  Low confidence (removed): {low_conf_count}")
            if list_to_remove:
                print(f"  Removed terms: {[t[4] for t in list_to_remove[:5]]}")


def extract_context(
    text: str,
    start: int,
    end: int,
    num_words_before: int = 5,
    num_words_after: int = 5
) -> str:
    """
    Extract context around a term (from original code)

    Args:
        text: Full text
        start: Start character index
        end: End character index
        num_words_before: Words to include before
        num_words_after: Words to include after

    Returns:
        Context string
    """
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
