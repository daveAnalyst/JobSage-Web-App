# ==================================================
# src/core_logic.py - Core Logic Functions
# (Refactored for Flask/Web App Usage)
# ==================================================
print("Loading core logic functions...") # Indicate module load

import re
import numpy as np
import pandas as pd
from sentence_transformers import util as st_util # Use alias to avoid confusion if flask.util is used
from scipy.stats import norm
import json
import random
from sklearn.metrics.pairwise import cosine_similarity # Alternative similarity calc

# --- Embedding Function ---
def get_text_embedding(text, embedding_model):
    """Generates sentence embedding for given text using the provided model."""
    if embedding_model is None:
        print("ERROR (get_text_embedding): Embedding model not provided.")
        return None
    if not isinstance(text, str): text = str(text)
    if not text.strip():
         print("Warning (get_text_embedding): Embedding empty string.")
         pass
    try:
        embedding = embedding_model.encode(text)
        return embedding
    except Exception as e:
        print(f"ERROR (get_text_embedding): Encoding failed - {e}")
        return None

# --- Question Generation ---
def generate_question(questions_df, gemini_model, category=None, difficulty=None, excluded_ids=[], max_examples=3):
    """
    Selects a predefined question matching criteria. Optionally attempts Gemini few-shot.

    Args:
        questions_df (pd.DataFrame): DataFrame of available questions.
        gemini_model: Initialized Gemini model instance.
        category (str, optional): Target category.
        difficulty (str, optional): Target difficulty.
        excluded_ids (list, optional): Question IDs to exclude.
        max_examples (int, optional): Max examples for few-shot prompt.

    Returns:
        tuple: (question_text, ideal_answer_points, skill_tags, question_id, difficulty) or error codes.
    """
    if questions_df.empty:
        print("ERROR (generate_question): Questions DataFrame is empty.")
        return "Error: Questions DataFrame is empty.", "N/A", "N/A", -3, "N/A"

    # --- Filter predefined questions ---
    filtered_questions = questions_df.copy()
    if category:
        filtered_questions = filtered_questions[filtered_questions['category'].str.lower() == category.lower()]
    if difficulty:
        filtered_questions = filtered_questions[filtered_questions['difficulty'].str.lower() == difficulty.lower()]
    if excluded_ids:
        filtered_questions = filtered_questions[~filtered_questions['question_id'].isin(excluded_ids)]

    if filtered_questions.empty: # Fallback logic
        print(f"Warning (generate_question): No questions found for Cat='{category}', Diff='{difficulty}'. Broadening search...")
        filtered_questions = questions_df.copy()
        if category: filtered_questions = filtered_questions[filtered_questions['category'].str.lower() == category.lower()]
        if excluded_ids: filtered_questions = filtered_questions[~filtered_questions['question_id'].isin(excluded_ids)]
        if filtered_questions.empty and difficulty:
             print(f"Broadening further: Ignoring category, keeping Diff='{difficulty}'...")
             filtered_questions = questions_df.copy()
             filtered_questions = filtered_questions[filtered_questions['difficulty'].str.lower() == difficulty.lower()]
             if excluded_ids: filtered_questions = filtered_questions[~filtered_questions['question_id'].isin(excluded_ids)]
        if filtered_questions.empty:
             print("Broadening further: Using any available question...")
             filtered_questions = questions_df[~questions_df['question_id'].isin(excluded_ids)]
             if filtered_questions.empty: return "No more unique questions available.", "End.", "End", -2, "End"

    # --- Select the question ---
    chosen_q = filtered_questions.sample(1).iloc[0]
    print(f"Selected question ID {chosen_q['question_id']} (Diff: {chosen_q['difficulty']}).")

    # --- Attempt Gemini generation (for capability demo only) ---
    if gemini_model:
        print("Attempting Gemini candidate generation (for demo purposes)...")
        # ... (rest of Gemini generation attempt logic - keep as is for demo) ...
        examples_for_prompt = questions_df.sample(min(max_examples, len(questions_df)))
        few_shot_prompt_text = "Example technical interview questions:\n\n"
        for _, row in examples_for_prompt.iterrows():
             few_shot_prompt_text += f"---\nCategory: {row['category']}\nDifficulty: {row['difficulty']}\nQuestion: {row['question_text']}\n---\n"
        target_cat_gen = category if category else random.choice(questions_df['category'].unique())
        target_diff_gen = difficulty if difficulty else random.choice(questions_df['difficulty'].unique())
        prompt = few_shot_prompt_text + f"\nGenerate one new, unique data science interview question.\nDesired Category: '{target_cat_gen}'\nDesired Difficulty: '{target_diff_gen}'\nRespond ONLY with the question text."
        try:
            safety_settings_med = [ {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
            gen_response = gemini_model.generate_content(prompt, safety_settings=safety_settings_med)
            generated_text = gen_response.text.strip()
            print(f"Gemini generated candidate: '{generated_text}' (Candidate NOT used).")
        except Exception as e: print(f"Gemini candidate generation failed: {e}.") # Log error but continue
    else:
        print("Info (generate_question): Gemini model unavailable for candidate generation demo.")

    # --- Return selected predefined question ---
    return chosen_q['question_text'], chosen_q['ideal_answer_points'], chosen_q['skill_tags'], chosen_q['question_id'], chosen_q['difficulty']

# --- LLM Score Parsing Helper ---
def parse_llm_score(response_text, scale=10):
    """Helper function to extract a numerical score from LLM text output."""
    # ... (keep the previous robust parsing logic) ...
    if not isinstance(response_text, str): return 0.0
    match = re.search(r'(\d+(\.\d+)?)\s*/\s*' + str(int(scale)), response_text)
    if match:
        try: return max(0.0, min(float(scale), float(match.group(1))))
        except ValueError: pass
    match = re.search(r'score[:\s]*?(\d+(\.\d+)?)\s*/\s*' + str(int(scale)), response_text, re.IGNORECASE)
    if match:
        try: return max(0.0, min(float(scale), float(match.group(1))))
        except ValueError: pass
    match = re.search(r'\b(?:score|rating|value)[:\s]*?(\d+(\.\d+)?)\b', response_text, re.IGNORECASE)
    if match:
         try: return max(0.0, min(float(scale), float(match.group(1))))
         except ValueError: pass
    match = re.search(r'\b(\d+(\.\d+)?)\b', response_text) # Find any number as last resort
    if match:
         try: return max(0.0, min(float(scale), float(match.group(1))))
         except ValueError: pass
    print(f"Warning (parse_llm_score): Could not parse score (scale/{scale}) from snippet: '{response_text[:100]}...'")
    return 0.0

# --- Answer Evaluation ---
def evaluate_answer(question_text, user_answer, ideal_answer_points, difficulty,
                    gemini_model, embedding_model, MAX_POINTS_PER_QUESTION=20):
    """Evaluates user answer using embeddings and Gemini LLM, scaling points by difficulty."""
    # Initialize defaults
    similarity_score = 0.0
    content_score_llm = 0.0
    clarity_score_llm = 0.0
    depth_score_llm = 0.0
    qualitative_feedback = "Evaluation unavailable."
    points = 0.0

    # 1. Similarity Score
    if embedding_model:
        user_embedding = get_text_embedding(user_answer, embedding_model) # Pass model
        ideal_embedding = get_text_embedding(ideal_answer_points, embedding_model) # Pass model
        if user_embedding is not None and ideal_embedding is not None:
            try:
                similarity_score = st_util.pytorch_cos_sim(user_embedding, ideal_embedding).item() * 10
                similarity_score = max(0.0, min(10.0, similarity_score))
            except Exception as e: print(f"Error calculating embedding similarity: {e}")
    else: print("ERROR (evaluate_answer): Embedding model not provided.")

    # 2. Gemini Evaluation
    if gemini_model:
        try:
            prompt = f"""Task: Evaluate the user's interview answer based on Content, Clarity, and Depth.
Instructions:
1. Content Score: Evaluate accuracy/correctness against expected points. Score 0-10. Provide 1 sentence rationale then 'Content Score: X/10'.
2. Clarity Score: Evaluate clear phrasing/structure (ignore accuracy). Score 0-10. Provide 1 sentence rationale then 'Clarity Score: X/10'.
3. Depth Score: Evaluate thoroughness/nuance beyond basics. Score 0-10. Provide 1 sentence rationale then 'Depth Score: X/10'.
4. Overall Feedback: Provide 2-3 sentences of concise, constructive overall feedback. Start with 'Overall Feedback:'.

Question: "{question_text}"
Expected Answer Points/Keywords: "{ideal_answer_points}"
Candidate's Answer: "{user_answer}"

Response Format (Use EXACT keywords 'Content Score:', 'Clarity Score:', 'Depth Score:', 'Overall Feedback:'):
Content Score: [score_float]/10. [Rationale]
Clarity Score: [score_float]/10. [Rationale]
Depth Score: [score_float]/10. [Rationale]
Overall Feedback: [Feedback text]
"""
            print("Requesting evaluation from Gemini...")
            safety_settings_med = [ {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
            response = gemini_model.generate_content(prompt, safety_settings=safety_settings_med)
            response_text = response.text

            # Parse scores
            content_match = re.search(r'Content Score[:\s]*?(\d+(\.\d+)?)\s*/\s*10', response_text, re.IGNORECASE)
            if content_match: content_score_llm = max(0.0, min(10.0, float(content_match.group(1))))
            else: content_score_llm = parse_llm_score(response_text, 10) # Fallback

            clarity_match = re.search(r'Clarity Score[:\s]*?(\d+(\.\d+)?)\s*/\s*10', response_text, re.IGNORECASE)
            if clarity_match: clarity_score_llm = max(0.0, min(10.0, float(clarity_match.group(1))))
            else: clarity_score_llm = parse_llm_score(response_text, 10)

            depth_match = re.search(r'Depth Score[:\s]*?(\d+(\.\d+)?)\s*/\s*10', response_text, re.IGNORECASE)
            if depth_match: depth_score_llm = max(0.0, min(10.0, float(depth_match.group(1))))
            else: depth_score_llm = parse_llm_score(response_text, 10)

            # Extract qualitative feedback
            feedback_match = re.search(r'Overall Feedback[:\s]*(.*)', response_text, re.IGNORECASE | re.DOTALL)
            if feedback_match: qualitative_feedback = feedback_match.group(1).strip()
            else: qualitative_feedback = response_text

            print("Gemini evaluation successful.")

        except Exception as e:
            error_msg = f"Error during Gemini evaluation: {e}"
            print(error_msg)
            qualitative_feedback = error_msg
            try: # Check block reason
                if response and response.prompt_feedback and response.prompt_feedback.block_reason:
                     qualitative_feedback += f" (Block Reason: {response.prompt_feedback.block_reason})"
            except: pass
            # Use similarity as content score if LLM fails
            content_score_llm = similarity_score
    else:
        print("Warning (evaluate_answer): Gemini model unavailable.")
        qualitative_feedback = "LLM evaluation skipped (model unavailable)."
        content_score_llm = similarity_score # Use similarity if no LLM

    # 3. Calculate Final Points
    base_points_ratio = (content_score_llm / 10 * 0.50 +
                         clarity_score_llm / 10 * 0.25 +
                         depth_score_llm / 10 * 0.25)
    difficulty_multiplier = {'easy': 0.9, 'medium': 1.0, 'hard': 1.1}
    difficulty_key = difficulty.lower() if isinstance(difficulty, str) else 'medium'
    scaled_points_ratio = base_points_ratio * difficulty_multiplier.get(difficulty_key, 1.0)
    points = scaled_points_ratio * MAX_POINTS_PER_QUESTION
    points = round(max(0, min(MAX_POINTS_PER_QUESTION, points)), 1)

    return {
        'similarity_score': round(similarity_score, 1),
        'content_score': round(content_score_llm, 1),
        'clarity_score': round(clarity_score_llm, 1),
        'depth_score': round(depth_score_llm, 1),
        'qualitative_feedback': qualitative_feedback,
        'points': points
    }

# --- Follow-up Question Generation ---
def generate_follow_up(question_text, user_answer, gemini_model):
    """Generates a relevant follow-up question using Gemini."""
    if gemini_model