# ==================================================
# src/core_logic.py - Core Logic Functions
# (Refactored for Flask/Web App Usage)
# ==================================================
print("Loading core logic functions...") # Indicate module load

import re
import numpy as np
import pandas as pd
from sentence_transformers import util as st_util # Use alias
from scipy.stats import norm
import json
import random
from sklearn.metrics.pairwise import cosine_similarity # Using this for job recs

# --- Embedding Function ---
def get_text_embedding(text, embedding_model):
    """
    Generates sentence embedding for given text using the provided model.

    Args:
        text (str): The text to embed.
        embedding_model: An initialized SentenceTransformer model instance.

    Returns:
        numpy.ndarray or None: The embedding vector, or None on error.
    """
    if embedding_model is None:
        print("ERROR (get_text_embedding): Embedding model not provided.")
        return None
    if not isinstance(text, str): text = str(text) # Ensure string input
    if not text.strip():
         print("Warning (get_text_embedding): Embedding empty string.")
         # Models typically handle empty strings, return their output
         # If specific handling needed (like zero vector), add here.
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
        gemini_model: Initialized Gemini model instance (can be None).
        category (str, optional): Target category.
        difficulty (str, optional): Target difficulty ('Easy', 'Medium', 'Hard').
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

    # --- Fallback Logic ---
    if filtered_questions.empty:
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
    # Use try-except for sampling just in case filtered_questions becomes unexpectedly empty after checks
    try:
        chosen_q = filtered_questions.sample(1).iloc[0]
        print(f"Selected question ID {chosen_q['question_id']} (Diff: {chosen_q['difficulty']}).")
    except ValueError: # Handles empty DataFrame sample error
         print("ERROR (generate_question): No eligible questions left after filtering.")
         return "No more unique questions available.", "End.", "End", -2, "End"


    # --- Attempt Gemini generation (for capability demo only) ---
    if gemini_model:
        print("Attempting Gemini candidate generation (for demo purposes)...")
        try:
            examples_for_prompt = questions_df.sample(min(max_examples, len(questions_df)))
            few_shot_prompt_text = "Example technical interview questions:\n\n"
            for _, row in examples_for_prompt.iterrows():
                 few_shot_prompt_text += f"---\nCategory: {row['category']}\nDifficulty: {row['difficulty']}\nQuestion: {row['question_text']}\n---\n"
            # Use chosen question's details if target category/difficulty weren't provided
            target_cat_gen = category if category else chosen_q['category']
            target_diff_gen = difficulty if difficulty else chosen_q['difficulty']
            prompt = few_shot_prompt_text + f"\nGenerate one new, unique data science interview question.\nDesired Category: '{target_cat_gen}'\nDesired Difficulty: '{target_diff_gen}'\nRespond ONLY with the question text."

            safety_settings_med = [ {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
            gen_response = gemini_model.generate_content(prompt, safety_settings=safety_settings_med)
            generated_text = gen_response.text.strip()
            print(f"Gemini generated candidate: '{generated_text}' (Candidate NOT used).")
        except Exception as e:
            print(f"Gemini candidate generation failed: {e}.")
            # Check for block reason if possible
            try:
                if gen_response and gen_response.prompt_feedback and gen_response.prompt_feedback.block_reason:
                     print(f"Generation blocked. Reason: {gen_response.prompt_feedback.block_reason}")
            except: pass # Ignore errors checking block reason
    else:
        print("Info (generate_question): Gemini model unavailable for candidate generation demo.")

    # --- Return selected predefined question ---
    return chosen_q['question_text'], chosen_q['ideal_answer_points'], chosen_q['skill_tags'], chosen_q['question_id'], chosen_q['difficulty']

# --- LLM Score Parsing Helper ---
def parse_llm_score(response_text, scale=10):
    """Helper function to extract a numerical score from LLM text output."""
    if not isinstance(response_text, str): return 0.0
    # Regex attempts... (keep robust version)
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
    match = re.search(r'\b(\d+(\.\d+)?)\b', response_text)
    if match:
         try: return max(0.0, min(float(scale), float(match.group(1))))
         except ValueError: pass
    print(f"Warning (parse_llm_score): Could not parse score (scale/{scale}) from snippet: '{response_text[:100]}...'")
    return 0.0

# --- Answer Evaluation ---
def evaluate_answer(question_text, user_answer, ideal_answer_points, difficulty,
                    gemini_model, embedding_model, MAX_POINTS_PER_QUESTION=20):
    """Evaluates user answer using embeddings and Gemini LLM, scaling points by difficulty."""
    similarity_score, content_score_llm, clarity_score_llm, depth_score_llm = 0.0, 0.0, 0.0, 0.0
    qualitative_feedback = "Evaluation unavailable."
    points = 0.0

    # 1. Similarity Score
    if embedding_model:
        user_embedding = get_text_embedding(user_answer, embedding_model)
        ideal_embedding = get_text_embedding(ideal_answer_points, embedding_model)
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
            else: content_score_llm = parse_llm_score(response_text, 10)

            clarity_match = re.search(r'Clarity Score[:\s]*?(\d+(\.\d+)?)\s*/\s*10', response_text, re.IGNORECASE)
            if clarity_match: clarity_score_llm = max(0.0, min(10.0, float(clarity_match.group(1))))
            else: clarity_score_llm = parse_llm_score(response_text, 10)

            depth_match = re.search(r'Depth Score[:\s]*?(\d+(\.\d+)?)\s*/\s*10', response_text, re.IGNORECASE)
            if depth_match: depth_score_llm = max(0.0, min(10.0, float(depth_match.group(1))))
            else: depth_score_llm = parse_llm_score(response_text, 10)

            # Extract qualitative feedback
            feedback_match = re.search(r'Overall Feedback[:\s]*(.*)', response_text, re.IGNORECASE | re.DOTALL)
            if feedback_match: qualitative_feedback = feedback_match.group(1).strip()
            else: qualitative_feedback = response_text # Fallback

            print("Gemini evaluation successful.")

        except Exception as e:
            error_msg = f"Error during Gemini evaluation: {e}"
            print(error_msg)
            qualitative_feedback = error_msg
            try: # Check block reason
                if response and response.prompt_feedback and response.prompt_feedback.block_reason:
                     qualitative_feedback += f" (Block Reason: {response.prompt_feedback.block_reason})"
            except: pass
            content_score_llm = similarity_score # Use similarity if LLM fails
    else:
        print("Warning (evaluate_answer): Gemini model unavailable.")
        qualitative_feedback = "LLM evaluation skipped (model unavailable)."
        content_score_llm = similarity_score

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
    if gemini_model is None:
        print("Info (generate_follow_up): Gemini model unavailable.")
        return None

    prompt = f"""Based on the original interview question and the user's answer provided below, ask ONE relevant and concise follow-up question (ending with '?'). Probe deeper or clarify a specific point. Ask only the question itself, without any preamble.

Original Question: "{question_text}"
User's Answer: "{user_answer}"

Follow-up Question:"""
    try:
        print("Requesting follow-up question from Gemini...")
        safety_settings_med = [ {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        response = gemini_model.generate_content(prompt, safety_settings=safety_settings_med)
        follow_up_question = response.text.strip()

        if not follow_up_question or len(follow_up_question) < 10: return None
        if not follow_up_question.endswith('?'): follow_up_question += "?"
        if "question:" in follow_up_question.lower() or "sorry" in follow_up_question.lower(): return None

        print(f"Gemini generated follow-up: {follow_up_question}")
        return follow_up_question
    except Exception as e:
        print(f"Error generating follow-up question with Gemini: {e}")
        try:
            if response and response.prompt_feedback and response.prompt_feedback.block_reason:
                 print(f"Gemini follow-up blocked. Reason: {response.prompt_feedback.block_reason}")
        except: pass
        return None

# --- Benchmarking Calculation ---
def calculate_benchmark(overall_session_score_10, mean, std_dev):
    """Calculates user's percentile benchmark against simulated norms (Score out of 10)."""
    if not isinstance(overall_session_score_10, (int, float)): return 0
    try:
        cdf_value = norm.cdf(overall_session_score_10, loc=mean, scale=std_dev)
        percentile_top = (1 - cdf_value) * 100
        return int(max(0, min(100, percentile_top)))
    except Exception as e: print(f"Error calculating benchmark: {e}"); return 0

# --- Study Topic Recommendation (RAG) ---
def recommend_study_topics(weakest_skills, study_rec_df):
    """Recommends study topics based on weakest skills using RAG lookup on study_rec_df."""
    recommendations = []
    if study_rec_df.empty: return recommendations
    if not isinstance(weakest_skills, list): return recommendations
    try:
        if 'skill_tag_lower' not in study_rec_df.columns:
             study_rec_df['skill_tag_lower'] = study_rec_df['skill_tag'].str.lower()
    except Exception as e: print(f"Error adding lowercase skill column: {e}"); return []

    for skill in weakest_skills:
        if not isinstance(skill, str): continue
        match = study_rec_df[study_rec_df['skill_tag_lower'] == skill.lower().strip()]
        if not match.empty:
            recommendations.append(match['recommendation_text'].iloc[0])
    # Consider removing the temporary column if this is the last use in the request lifecycle
    # study_rec_df = study_rec_df.drop(columns=['skill_tag_lower'], errors='ignore')
    return list(set(recommendations))

# --- CV Skill Extraction Fallback (Keywords) ---
def extract_cv_skills_keyword_fallback(cv_text):
    """Fallback: Extracts skills using a predefined keyword list."""
    print("Using keyword fallback for skill extraction.")
    if not isinstance(cv_text, str): return []
    # Using comprehensive list
    keywords = [ "python", "sql", "java", "c++", "c#", "r", "scala", "javascript", "typescript", "php", "swift", "kotlin", "go", "ruby", "perl", "bash", "powershell", "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly", "bokeh", "scikit-learn", "sklearn", "tensorflow", "keras", "pytorch", "torch", "jax", "theano", "caffe", "xgboost", "lightgbm", "catboost", "statsmodels", "nltk", "spacy", "gensim", "hugging face", "transformers", "opencv", "pillow", "aws", "azure", "gcp", "google cloud", "amazon web services", "cloud computing", "hadoop", "spark", "pyspark", "mapreduce", "hive", "pig", "impala", "kafka", "rabbitmq", "flink", "storm", "docker", "kubernetes", "k8s", "openshift", "terraform", "ansible", "ci/cd", "jenkins", "gitlab ci", "postgresql", "mysql", "sqlite", "sql server", "oracle", "mongodb", "cassandra", "redis", "neo4j", "elasticsearch", "nosql", "database design", "data modeling", "data warehousing", "etl", "tableau", "power bi", "qlik", "looker", "d3.js", "excel", "statistics", "probability", "econometrics", "calculus", "linear algebra", "discrete math", "machine learning", "ml", "deep learning", "dl", "artificial intelligence", "ai", "natural language processing", "nlp", "computer vision", "cv", "speech recognition", "reinforcement learning", "rl", "data analysis", "data mining", "predictive modeling", "forecasting", "optimization", "operations research", "a/b testing", "experiment design", "causal inference", "algorithms", "data structures", "object-oriented programming", "oop", "functional programming", "system design", "distributed systems", "microservices", "api design", "rest", "graphql", "communication", "presentation", "leadership", "teamwork", "collaboration", "problem-solving", "critical thinking", "agile", "scrum", "project management", "product management" ]
    cv_lower = cv_text.lower()
    found_skills = [k for k in keywords if re.search(r'\b' + re.escape(k) + r'\b', cv_lower)]
    return sorted(list(set(found_skills)))

# --- CV Skill Extraction (Using Gemini) ---
def extract_cv_skills(cv_text, gemini_model):
    """Extracts skills from a CV using Gemini API."""
    if gemini_model is None:
        print("Error (extract_cv_skills): Gemini model unavailable.")
        return extract_cv_skills_keyword_fallback(cv_text)
    if not isinstance(cv_text, str) or not cv_text.strip(): return []

    prompt = f"""Analyze the following CV text and extract a comprehensive list of professional skills. Include programming languages, libraries, frameworks, tools, platforms (cloud), methodologies, core concepts (e.g., machine learning, statistics), and relevant soft skills (e.g., communication).
Format the output strictly as a JSON array of unique strings. Example: ["Python", "SQL", "Scikit-learn", "AWS", "Agile", "Communication"]

CV Text:
---
{cv_text}
---

JSON Skill Array:"""
    try:
        print("Requesting CV skill extraction from Gemini...")
        safety_settings_med = [ {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        response = gemini_model.generate_content(prompt, safety_settings=safety_settings_med)
        cleaned_response_text = re.sub(r'```json\s*([\s\S]*?)\s*```', r'\1', response.text, flags=re.IGNORECASE)
        skills = json.loads(cleaned_response_text)
        if isinstance(skills, list) and all(isinstance(s, str) for s in skills):
            print(f"Gemini extracted {len(skills)} skills.")
            return sorted(list(set(skills)))
        else:
            print("Warning: Gemini skill extraction did not return a valid list of strings.")
            return extract_cv_skills_keyword_fallback(cv_text)
    except json.JSONDecodeError as e:
         print(f"Error parsing Gemini JSON skill response: {e}")
         print(f"Raw Response: {response.text[:500] if 'response' in locals() else 'N/A'}")
         return extract_cv_skills_keyword_fallback(cv_text)
    except Exception as e:
        print(f"Error extracting skills with Gemini: {e}")
        try: # Check block reason
            if response and response.prompt_feedback and response.prompt_feedback.block_reason:
                 print(f"Skill extraction blocked. Reason: {response.prompt_feedback.block_reason}")
        except: pass
        return extract_cv_skills_keyword_fallback(cv_text)


# --- Job Recommendation (RAG via Embedding Similarity) ---
def recommend_jobs(cv_skills_embedding, jobs_df, top_n=3):
    """Recommends jobs based on cosine similarity between CV embedding and pre-computed job embeddings."""
    if cv_skills_embedding is None: return []
    # Ensure jobs_df is a DataFrame and has the required columns
    if not isinstance(jobs_df, pd.DataFrame) or jobs_df.empty or 'embeddings' not in jobs_df.columns or jobs_df['embeddings'].isnull().all():
        print("Warning (recommend_jobs): Invalid or empty jobs_df provided.")
        return []

    valid_jobs_df = jobs_df.dropna(subset=['embeddings']).copy()
    if valid_jobs_df.empty: return []

    try: job_embeddings = np.stack(valid_jobs_df['embeddings'].values)
    except ValueError as e: # Handle potential errors if embeddings are not uniform
        print(f"Error stacking job embeddings (check consistency): {e}"); return []
    except Exception as e: print(f"Error stacking job embeddings: {e}"); return []


    try:
        if not isinstance(cv_skills_embedding, np.ndarray): cv_skills_embedding = np.array(cv_skills_embedding)
        # Use sklearn cosine similarity for potentially better handling of shapes
        similarities = cosine_similarity(cv_skills_embedding.reshape(1, -1), job_embeddings)[0]

        valid_jobs_df['similarity_score'] = similarities
        actual_top_n = min(top_n, len(valid_jobs_df))
        top_jobs = valid_jobs_df.nlargest(actual_top_n, 'similarity_score')
        # Select desired columns for output
        recommended_jobs = top_jobs[['title', 'company', 'link', 'similarity_score']].to_dict('records')
        # Round score for cleaner output
        for job in recommended_jobs: job['similarity_score'] = round(job['similarity_score'], 3)
        return recommended_jobs
    except Exception as e: print(f"Error during job recommendation similarity calculation: {e}"); return []

# --- Negotiation Simulator ---
def simulate_negotiation(job_title, years_experience, current_salary, gemini_model):
    """Simulates a salary negotiation conversation using Gemini."""
    if gemini_model is None: return "Negotiation simulation unavailable (Model not loaded)."
    try: # Input validation
        years_exp_float = float(years_experience); current_salary_float = float(current_salary)
        if not isinstance(job_title, str) or not job_title.strip(): raise ValueError("Job Title missing")
    except (ValueError, TypeError) as e: return f"Error: Invalid input. Details: {e}"

    # Basic salary estimation
    base_multiplier = 1.2 if isinstance(job_title, str) and job_title.lower() in ["data scientist", "ml engineer", "research scientist"] else 1.0
    experience_factor = min(years_exp_float * 0.1, 0.5)
    target_min = int(current_salary_float * (1.1 + base_multiplier * experience_factor))
    target_max = int(current_salary_float * (1.2 + base_multiplier * experience_factor))

    prompt = f"""Simulate a brief, realistic salary negotiation dialogue (4-6 conversational turns total) for a '{job_title}' position.

Candidate Profile:
- Current approximate salary: ${current_salary_float:,.0f}
- Years of relevant experience: {years_exp_float:.1f}
- Reasonable target salary range based on profile: ${target_min:,.0f} - ${target_max:,.0f}

Instructions for Dialogue:
1. Start with the Hiring Manager making an initial verbal offer.
2. Include realistic back-and-forth. The 'Candidate' should express gratitude, possibly counter-offer, ask clarifying questions, or discuss other benefits.
3. Conclude the dialogue.

Instructions for Feedback:
4. After the dialogue, provide exactly 3 specific, actionable feedback points analyzing the CANDIDATE's negotiation strategy shown ONLY in the dialogue you generated. Start this section clearly with '--- FEEDBACK ---'.

Dialogue & Feedback:"""
    try:
        print(f"Requesting negotiation simulation from Gemini (Target: ${target_min:,.0f}-${target_max:,.0f})...")
        safety_settings_med = [ {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        response = gemini_model.generate_content(prompt, safety_settings=safety_settings_med)
        if response.text and len(response.text) > 70 and "FEEDBACK" in response.text:
             print("Negotiation simulation generated."); return response.text.strip()
        else:
             block_reason = "Response invalid/short"; response_text_snippet = response.text[:100] if response.text else 'Empty'
             try:
                 if response and response.prompt_feedback and response.prompt_feedback.block_reason: block_reason = response.prompt_feedback.block_reason
             except: pass
             print(f"Negotiation sim seems invalid/blocked (Reason: {block_reason}). Resp: {response_text_snippet}")
             return "Error: Could not generate valid negotiation simulation."
    except Exception as e: print(f"Error during negotiation simulation: {e}"); return f"Error: {e}"

# --- Resume Tweak Suggestion ---
def suggest_resume_tweaks(weakest_skills, cv_text, gemini_model):
    """Suggests potential resume improvements based on identified weak skills using Gemini."""
    if gemini_model is None: return ["Resume tweak suggestions unavailable (Model not loaded)."]
    if not weakest_skills: return []
    if not isinstance(cv_text, str) or len(cv_text) < 50: return ["CV text too short/invalid."]

    skills_to_focus = weakest_skills[:2]
    tweaks = []
    print(f"Requesting resume tweak suggestions from Gemini for skills: {skills_to_focus}...")
    try:
        for skill in skills_to_focus:
            prompt = f"""Analyze the following CV. The candidate showed weakness in the skill '{skill}' during a mock interview.
Provide ONE specific, actionable suggestion (1-2 sentences) for how the candidate could improve their CV to better showcase potential experience or knowledge related to '{skill}'. If the skill seems completely absent and unaddressable, state that politely.

CV Text:
---
{cv_text}
---

Suggestion for '{skill}':"""
            safety_settings_med = [ {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
            response = gemini_model.generate_content(prompt, safety_settings=safety_settings_med)
            tweak = response.text.strip()
            if tweak and len(tweak) > 10: tweaks.append(tweak)
            else: print(f"Gemini provided no valid tweak for {skill}. Response: {tweak[:100]}")
        print("Resume tweak suggestions generated.")
        return tweaks
    except Exception as e:
        print(f"Error generating resume tweaks with Gemini: {e}")
        try: # Check block reason
            if response and response.prompt_feedback and response.prompt_feedback.block_reason:
                 print(f"Tweak generation blocked. Reason: {response.prompt_feedback.block_reason}")
        except: pass
        return ["Error occurred generating resume suggestions."]


print("-" * 30)
print("CORE LOGIC FUNCTIONS DEFINED (Refactored for Flask)")
print("-" * 30)