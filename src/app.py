# ==========================================
# src/app.py - Main Flask Application
# (Corrected for JSON Serialization in Session)
# ==========================================
import os
import pandas as pd
import numpy as np
import json
import re
import random
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util as st_util # Alias util
from scipy.stats import norm
import traceback

# --- Load Environment Variables ---
load_dotenv()

# --- Initialize Flask App ---
app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-insecure-change-me!')
if app.secret_key == 'dev-secret-key-insecure-change-me!':
    print("WARNING: Using default Flask secret key. Set FLASK_SECRET_KEY in .env!")

# --- Initialize AI Models ---
print("Initializing AI models...")
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
gemini_model = None
embedding_model = None

# Initialize Gemini
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        print("Gemini model initialized.")
    except Exception as e: print(f"ERROR initializing Gemini: {e}")
else: print("Warning: GEMINI_API_KEY not found.")

# Initialize Embedding Model
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedding model loaded.")
except Exception as e_emb: print(f"ERROR loading embedding model: {e_emb}")

# --- Define and Load Data In-Memory ---
print("Defining and loading data into memory...")
try:
    # 1. Questions Data
    questions_data = {
        'question_id': [1, 2, 3, 4, 5],
        'question_text': [ "Explain L1/L2 regularization.", "Handle missing data?", "Design A/B test framework?", "Explain bias-variance tradeoff.", "SQL query: top 3 dept avg salary?"],
        'ideal_answer_points': ["L1=Lasso, sparsity, feature selection. L2=Ridge, smaller weights, multicollinearity.", "Deletion (MCAR/small), Imputation (mean/median/mode/model), Indicator vars, Algo support. Depends on pattern/amount.", "Define metric, power analysis->sample size, randomize groups, track, run duration, analyze (t-test/chi2), check validity.", "Bias=underfitting (simple model). Variance=overfitting (complex model). Goal=balance for generalization.", "SELECT d.dept_name, AVG(e.salary) as avg_sal FROM emp e JOIN dept d ON e.dept_id = d.dept_id GROUP BY d.dept_name ORDER BY avg_sal DESC LIMIT 3;"],
        'skill_tags': ["Machine Learning,Regularization", "Data Preprocessing", "Experimental Design,Statistics", "Machine Learning,Evaluation", "SQL,Data Analysis"],
        'category': ["Machine Learning", "Data Preprocessing", "Experimental Design", "Machine Learning", "SQL"],
        'difficulty': ["Medium", "Easy", "Hard", "Medium", "Medium"]
    }
    questions_df = pd.DataFrame(questions_data)
    print(f"Loaded {len(questions_df)} questions.")

    # 2. Jobs Data
    jobs_data = {
        'job_id': [1, 2, 3, 4, 5], 'title': ["Data Scientist", "ML Engineer", "Data Analyst", "Research Scientist", "Data Engineer"],
        'company': ["TechCorp", "AIStartup", "FinTech Inc", "PharmaLabs", "BigData Co"],
        'description': ["Build recommendation algorithms using Python, SQL, ML.", "Deploy CV models using PyTorch, TensorFlow, MLOps.", "Analyze financial data with SQL, Excel, Tableau.", "Analyze clinical trial data using R, Statistics.", "Build big data pipelines using Spark, Hadoop, SQL, Python, Cloud."],
        'link': ["#job1", "#job2", "#job3", "#job4", "#job5"],
        'skills_required': [ ["Python", "SQL", "ML", "Stats", "A/B Test"], ["PyTorch", "TF", "CV", "MLOps", "Python"], ["SQL", "Excel", "Tableau", "Stats"], ["R", "Stats", "Exp Design"], ["Spark", "Hadoop", "SQL", "Python", "Cloud"] ]
    }
    jobs_df = pd.DataFrame(jobs_data)
    # Pre-compute Job Embeddings
    jobs_df['embeddings'] = None
    if embedding_model and 'description' in jobs_df.columns:
        try:
            jobs_df['embeddings'] = jobs_df['description'].fillna('').apply(lambda x: embedding_model.encode(x) if x else None)
            print(f"Job embeddings computed for {jobs_df['embeddings'].notna().sum()} jobs.")
            jobs_df = jobs_df.dropna(subset=['embeddings'])
        except Exception as e: print(f"Error computing job embeddings: {e}")
    else: print("Warning: Cannot compute job embeddings.")
    print(f"Loaded {len(jobs_df)} jobs.")

    # 3. Study Recommendations Data
    study_rec_data_list = []
    temp_study_data = [ {"category": "Machine Learning", "resources": [{"title": "ML Mastery Blog", "type": "Blog"}, {"title": "PRML Book", "type": "Book"}]}, {"category": "SQL", "resources": [{"title": "SQLZoo Practice", "type": "Practice"}, {"title": "Leetcode DB", "type": "Practice"}]}, {"category": "Data Preprocessing", "resources": [{"title": "Feature Eng Book", "type": "Book"}, {"title": "Kaggle Data Cleaning Course", "type": "Course"}]}, {"category": "Experimental Design", "resources": [{"title": "Trustworthy Exp Book", "type": "Book"}, {"title": "Udacity A/B Course", "type": "Course"}]}, {"category": "Statistics", "resources": [{"title": "StatQuest YouTube", "type": "Video"}, {"title": "Practical Stats Book", "type": "Book"}]} ]
    for item in temp_study_data:
         category = item['category']; rec_text = f"For {category}, check out: " + ", ".join([f"{res['title']} ({res['type']})" for res in item['resources']])
         study_rec_data_list.append({'skill_tag': category, 'recommendation_text': rec_text})
    study_rec_df = pd.DataFrame(study_rec_data_list)
    if not study_rec_df.empty: study_rec_df['skill_tag_lower'] = study_rec_df['skill_tag'].str.lower()
    print(f"Loaded {len(study_rec_df)} study recommendations.")

    # 4. Negotiation Scenarios Data
    negotiation_data = { 'scenario_id': [1, 2, 3, 4], 'role_type': ['FAANG', 'Startup', 'FAANG', 'Startup'], 'recruiter_statement': ["Offer: $120k base.", "Offer: $90k base, 0.1% equity.", "Offer: $135k base.", "Offer: $100k base."], 'good_user_counter_example': ["Thanks! Expected closer to $135k based on research/other offers. Flexible?", "Appreciate it. Given market/experience, targeting $105k base. Vesting details?", "Strong offer, thanks. Believe value aligns closer to $145k-$150k. Sign-on/perf bonus negotiable?", "Thanks! Bonus structure details? Aiming for $120k total comp."], 'feedback_hint': ["Countered reasonably.", "Acknowledged equity, target base.", "Justified higher range.", "Asked clarifying Qs."] }
    negotiation_scenarios_df = pd.DataFrame(negotiation_data)
    print(f"Loaded {len(negotiation_scenarios_df)} negotiation scenarios.")

    # 5. Leaderboard Data
    leaderboard_data = {'Rank': [1, 2, 3, 4, 5], 'User': ["AI_Legend", "CodeNinja", "DataGuru", "StatsWizard", "ProbSolver"], 'Score': [9.8, 9.5, 9.1, 8.8, 8.5], 'Badges': ["Passed!,Top Performer", "Passed!,Top Performer", "Passed!", "Passed!", "Passed!"]}
    leaderboard_df = pd.DataFrame(leaderboard_data).set_index('Rank')
    print(f"Loaded {len(leaderboard_df)} leaderboard entries.")

except Exception as e:
    print(f"An unexpected error occurred defining/loading data: {e}")
    questions_df = pd.DataFrame()
    jobs_df = pd.DataFrame(columns=['embeddings'])
    study_rec_df = pd.DataFrame()
    negotiation_scenarios_df = pd.DataFrame()
    leaderboard_df = pd.DataFrame()
# --- End Data Loading ---

# --- Import Core Logic Module ---
try:
    import core_logic # Assumes src/core_logic.py exists and is importable
    print("Core logic module imported successfully.")
except ModuleNotFoundError:
     print("ERROR: Could not find core_logic.py.")
     core_logic = None
except Exception as e: # Catch other import errors like syntax errors
     print(f"ERROR importing core_logic: {e}")
     core_logic = None

print("Initialization complete.")

# --- Define App-Level Constants ---
APP_FAANG_MEAN_SCORE = float(os.getenv('FAANG_MEAN_SCORE', 7.5))
APP_FAANG_STD_DEV = float(os.getenv('FAANG_STD_DEV', 1.0))
APP_PASSING_THRESHOLD = float(os.getenv('PASSING_THRESHOLD', 7.0)) # Score out of 10
APP_MAX_POINTS_PER_QUESTION = int(os.getenv('MAX_POINTS_PER_QUESTION', 20))
print("App constants defined.")
# --- End Constants ---

# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the main HTML page."""
    session.clear()
    print("Rendering index page, clearing session.")
    return render_template('index.html')

@app.route('/start_interview', methods=['POST'])
def start_interview():
    endpoint_name = '/start_interview'
    print(f"Received {endpoint_name} request")
    if not core_logic or not hasattr(core_logic, 'generate_question'):
        return jsonify({"error": "Core logic not loaded properly"}), 500

    session.clear()

    try:
        cv_text = request.json.get('cv_text', '')
        extracted_skills = []
        cv_embedding_list = None
        if cv_text:
            if hasattr(core_logic, 'extract_cv_skills') and gemini_model:
                 extracted_skills = core_logic.extract_cv_skills(cv_text, gemini_model)
            if embedding_model and hasattr(core_logic, 'get_text_embedding'):
                 cv_emb = core_logic.get_text_embedding(cv_text, embedding_model)
                 if cv_emb is not None: cv_embedding_list = cv_emb.tolist() # Convert numpy array

        # Initialize session state
        session['interview_history'] = []
        session['asked_question_ids'] = [] # Will store standard python ints
        session['total_session_points'] = 0.0
        session['current_difficulty'] = "Medium"
        session['num_questions_total'] = 3
        session['cv_skills'] = extracted_skills # list of strings
        session['cv_embedding'] = cv_embedding_list # list of floats
        session['cv_text_internal'] = cv_text # string

        # Get the first question
        q_text, q_ideal, q_skills, q_id, q_difficulty = core_logic.generate_question(
            questions_df=questions_df, gemini_model=gemini_model,
            difficulty=session['current_difficulty'],
            excluded_ids=session['asked_question_ids']
        )
        if q_id < 0: return jsonify({"error": "Could not generate first question."}), 500

        # Store JSON serializable types in session
        session['current_question'] = {
            'id': int(q_id), # <<< Cast to python int
            'text': q_text, 'ideal': q_ideal, 'skills': q_skills, 'difficulty': q_difficulty
        }
        session['asked_question_ids'] = session.get('asked_question_ids', []) + [int(q_id)] # <<< Cast to python int
        session.modified = True

        print(f"Started interview, first question ID: {q_id}")
        return jsonify({
            "question_number": 1, "total_questions": session['num_questions_total'],
            "question_text": q_text, "category": q_skills, "difficulty": q_difficulty
        })
    except Exception as e:
        print(f"Error in {endpoint_name}: {e}"); traceback.print_exc()
        return jsonify({"error": f"An error occurred: {e}"}), 500

@app.route('/submit_answer', methods=['POST'])
def submit_answer_route():
    endpoint_name = '/submit_answer'
    print(f"Received {endpoint_name} request")
    if not core_logic or not hasattr(core_logic, 'evaluate_answer') or not hasattr(core_logic, 'generate_follow_up'):
        return jsonify({"error": "Core logic functions not available."}), 500

    try:
        data = request.get_json()
        user_answer = data.get('answer')
        current_q = session.get('current_question')

        if not current_q or not isinstance(user_answer, str):
            return jsonify({"error": "Missing answer or question context in session"}), 400

        print(f"Evaluating answer for QID {current_q.get('id')}")
        evaluation = core_logic.evaluate_answer(
            question_text=current_q.get('text'), user_answer=user_answer,
            ideal_answer_points=current_q.get('ideal'), difficulty=current_q.get('difficulty'),
            gemini_model=gemini_model, embedding_model=embedding_model,
            MAX_POINTS_PER_QUESTION=APP_MAX_POINTS_PER_QUESTION
        )

        # Ensure evaluation results are serializable before adding to history
        serializable_evaluation = json.loads(json.dumps(evaluation, default=str)) # Basic way to force serialization

        session['total_session_points'] = session.get('total_session_points', 0.0) + evaluation.get('points', 0.0)
        history_item = {'q_id': current_q.get('id'), 'difficulty': current_q.get('difficulty'), 'skills': current_q.get('skills'), 'eval': serializable_evaluation}
        session['interview_history'] = session.get('interview_history', []) + [history_item]

        follow_up_question = core_logic.generate_follow_up(
            question_text=current_q.get('text'), user_answer=user_answer, gemini_model=gemini_model
        )

        # --- Get NEXT question or signal end ---
        next_question_data = {}
        current_idx = len(session.get('interview_history', []))
        total_questions = session.get('num_questions_total', 3)

        if current_idx >= total_questions:
            print("Interview complete.")
            session['interview_complete'] = True
        else:
            print("Generating next question...")
            last_points = evaluation.get('points', 0.0)
            current_diff = session.get('current_difficulty', 'Medium')
            next_difficulty = current_diff
            # ... (difficulty adaptation logic) ...
            upper_threshold = APP_MAX_POINTS_PER_QUESTION * 0.8
            lower_threshold = APP_MAX_POINTS_PER_QUESTION * 0.5
            if last_points >= upper_threshold:
                 if current_diff == "Easy": next_difficulty = "Medium"
                 elif current_diff == "Medium": next_difficulty = "Hard"
            elif last_points < lower_threshold:
                 if current_diff == "Hard": next_difficulty = "Medium"
                 elif current_diff == "Medium": next_difficulty = "Easy"
            session['current_difficulty'] = next_difficulty

            nq_text, nq_ideal, nq_skills, nq_id, nq_difficulty = core_logic.generate_question(
                questions_df=questions_df, gemini_model=gemini_model,
                difficulty=next_difficulty, excluded_ids=session.get('asked_question_ids', [])
            )
            if nq_id < 0:
                 print("Could not get next question or finished.")
                 session['interview_complete'] = True
            else:
                 # Store JSON serializable types <<< FIX AREA
                 session['current_question'] = {'id': int(nq_id), 'text': nq_text, 'ideal': nq_ideal, 'skills': nq_skills, 'difficulty': nq_difficulty} # Cast id
                 session['asked_question_ids'] = session.get('asked_question_ids', []) + [int(nq_id)] # Cast id
                 next_question_data = {
                     "question_number": current_idx + 1, "total_questions": total_questions,
                     "question_text": nq_text, "category": nq_skills, "difficulty": nq_difficulty
                 }

        session.modified = True

        # Return serializable data
        return jsonify({
            "evaluation": serializable_evaluation, # Return the serializable version
            "follow_up": follow_up_question,
            "next_question": next_question_data if not session.get('interview_complete') else None,
            "interview_complete": session.get('interview_complete', False),
            "current_total_points": session.get('total_session_points', 0.0)
        })

    except Exception as e:
        print(f"Error in {endpoint_name}: {e}"); traceback.print_exc()
        return jsonify({"error": f"An error occurred: {e}"}), 500

# --- simulate_negotiation_route ---
@app.route('/simulate_negotiation', methods=['POST'])
def simulate_negotiation_route():
    endpoint_name = '/simulate_negotiation'
    print(f"Received {endpoint_name} request")
    if not core_logic or not hasattr(core_logic, 'simulate_negotiation'):
        return jsonify({"error": "Core logic not loaded properly"}), 500

    try:
        data = request.get_json()
        job_title = data.get('job_title')
        years_experience = data.get('years_experience')
        current_salary = data.get('current_salary')

        simulation_result = core_logic.simulate_negotiation(
            job_title=job_title, years_experience=years_experience,
            current_salary=current_salary, gemini_model=gemini_model
        )
        return jsonify({"simulation_text": simulation_result})

    except Exception as e:
        print(f"Error in {endpoint_name}: {e}"); traceback.print_exc()
        return jsonify({"error": f"An error occurred: {e}"}), 500

# --- get_results_route ---
@app.route('/get_results', methods=['GET'])
def get_results_route():
    endpoint_name = '/get_results'
    print(f"Received {endpoint_name} request")
    if not core_logic or not all([hasattr(core_logic, name) for name in ['calculate_benchmark', 'recommend_study_topics', 'recommend_jobs', 'suggest_resume_tweaks']]):
          return jsonify({"error": "Core logic analysis functions not available"}), 500

    if not session.get('interview_complete'):
        return jsonify({"message": "Interview not yet complete. Finish submitting answers."}), 400 # Use 400 Bad Request

    try:
        history = session.get('interview_history', [])
        cv_skills = session.get('cv_skills', [])
        cv_embedding_list = session.get('cv_embedding')
        cv_embedding = np.array(cv_embedding_list) if cv_embedding_list is not None else None # Check for None
        cv_text_internal = session.get('cv_text_internal', '')

        # Calculate overall score (Avg Similarity)
        all_similarity_scores = [item['eval'].get('similarity_score', 0.0) for item in history]
        overall_score = float(np.mean(all_similarity_scores)) if all_similarity_scores else 0.0 # Cast to float

        benchmark = core_logic.calculate_benchmark(overall_score, APP_FAANG_MEAN_SCORE, APP_FAANG_STD_DEV)

        # Skill analysis & Study Recs
        session_skills_perf = {}
        for item in history:
             eval_data = item.get('eval', {})
             sim_score = eval_data.get('similarity_score', 0.0)
             skills_str = item.get('skills', '')
             if skills_str and isinstance(skills_str, str):
                  for skill in skills_str.split(','):
                       skill = skill.strip().lower();
                       if not skill: continue
                       if skill not in session_skills_perf: session_skills_perf[skill] = []
                       session_skills_perf[skill].append(sim_score)
        # Ensure avg score is float
        skill_avg_scores = {skill: float(round(np.mean(scores), 1)) for skill, scores in session_skills_perf.items() if scores}
        weakest_skills = [skill for skill, avg_score in skill_avg_scores.items() if avg_score < 6.5]
        study_recs = core_logic.recommend_study_topics(weakest_skills, study_rec_df)

        # Resume Tweaks
        resume_tweaks = core_logic.suggest_resume_tweaks(weakest_skills, cv_text_internal, gemini_model)

        # Job Recs
        job_recs = []
        if overall_score >= APP_PASSING_THRESHOLD and cv_embedding is not None:
             job_recs = core_logic.recommend_jobs(cv_embedding, jobs_df, top_n=3)

        # Award Badges
        earned_badges = set()
        if overall_score >= APP_PASSING_THRESHOLD: earned_badges.add("Passed!")
        total_points = float(session.get('total_session_points', 0.0)) # Cast to float
        total_possible_points = session.get('num_questions_total', 3) * APP_MAX_POINTS_PER_QUESTION
        if total_possible_points > 0 and total_points >= (total_possible_points * 0.8): earned_badges.add("High Scorer!")

        results = {
            "overall_score": round(overall_score, 1), "total_points": round(total_points, 1),
            "benchmark_percentile": benchmark, # Already int
            "skill_performance": skill_avg_scores, # Dict of str:float is serializable
            "weakest_skills": weakest_skills, # List of str
            "study_recommendations": study_recs, # List of str
            "resume_tweaks": resume_tweaks, # List of str
            "job_recommendations": job_recs, # List of dicts (should be serializable if df was)
            "badges_earned": sorted(list(earned_badges)) # List of str
        }
        print(f"Returning results for completed interview.")
        return jsonify(results)

    except Exception as e:
         print(f"Error in {endpoint_name}: {e}"); traceback.print_exc()
         return jsonify({"error": f"An error occurred getting results: {e}"}), 500


# --- Run the App ---
if __name__ == '__main__':
    print("Starting Flask development server via app.py...")
    port = int(os.getenv('PORT', 5000))
    debug_mode = os.getenv('FLASK_ENV', 'development').lower() == 'development'
    print(f"Running in {'Debug' if debug_mode else 'Production'} mode on port {port}")
    # Use threaded=False if experiencing issues with models/state in debug mode reload
    app.run(debug=debug_mode, host='0.0.0.0', port=port, threaded=True)