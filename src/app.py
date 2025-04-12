# ==========================================
# src/app.py - Initial Flask Setup & Data
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
from sentence_transformers import SentenceTransformer, util
from scipy.stats import norm
# Import necessary functions from core_logic (we'll create this file next)
# For now, we might define placeholder functions here or wait until core_logic is populated

# --- Load Environment Variables ---
load_dotenv() # Loads variables from .env file

# --- Initialize Flask App ---
app = Flask(__name__, template_folder='../templates', static_folder='../static') # Point to correct folders relative to src/
# IMPORTANT: Set a real secret key in your .env file
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-please-change')
if app.secret_key == 'dev-secret-key-please-change':
    print("WARNING: Using default Flask secret key. Set FLASK_SECRET_KEY in .env for production!")

# --- Initialize AI Models ---
print("Initializing AI models...")
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
gemini_model = None
embedding_model = None

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        print("Gemini model initialized.")
    except Exception as e:
        print(f"ERROR initializing Gemini: {e}")
else:
    print("Warning: GEMINI_API_KEY not found in environment.")

try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedding model loaded.")
except Exception as e:
    print(f"ERROR loading embedding model: {e}")

# --- Define and Load Data In-Memory ---
print("Defining and loading data into memory...")

# 1. Questions Data
questions_data = {
    'question_id': [1, 2, 3, 4, 5],
    'question_text': [
        "Explain the difference between L1 and L2 regularization in machine learning models.",
        "How would you handle missing data in a dataset?",
        "Design an A/B testing framework for a new feature on an e-commerce website.",
        "Explain the bias-variance tradeoff in machine learning.",
        "Write a SQL query to find the top 3 departments with the highest average salary."
    ],
    'ideal_answer_points': [
        "L1 regularization (Lasso) adds absolute coefficient values to loss, promoting sparsity/feature selection. L2 regularization (Ridge) adds squared coefficient values, shrinking coefficients but keeping most non-zero, handling multicollinearity.",
        "Methods include Deletion (listwise/pairwise if MCAR/small loss), Imputation (mean/median/mode - simple, affects variance; regression/KNN - more accurate), using algorithms tolerant to NaNs, or creating indicator variables. Choice depends on missingness pattern (MCAR/MAR/MNAR) and data amount.",
        "Define clear metrics (conversion, revenue). Calculate sample size via power analysis. Randomize users into control/treatment. Implement tracking. Run for fixed duration (e.g., 2 weeks). Analyze with statistical tests (t-test/chi-square). Consider novelty effects, seasonality, contamination.",
        "Tradeoff between model simplicity (high bias/underfitting - misses patterns) and complexity (high variance/overfitting - fits noise). Goal is optimal balance minimizing total error (bias^2 + variance + irreducible error) for good generalization.",
        "SELECT d.department_name, AVG(e.salary) as avg_salary FROM employees e JOIN departments d ON e.department_id = d.department_id GROUP BY d.department_name ORDER BY avg_salary DESC LIMIT 3;"
    ],
    'skill_tags': [
        "Machine Learning,Regularization,Model Tuning",
        "Data Preprocessing,Data Cleaning",
        "Experimental Design,Statistics,A/B Testing",
        "Machine Learning,Model Tuning,Evaluation",
        "SQL,Data Analysis"
    ],
     'category': [
        "Machine Learning", "Data Preprocessing", "Experimental Design", "Machine Learning", "SQL"
     ],
     'difficulty': [
        "Medium", "Easy", "Hard", "Medium", "Medium"
     ]
}
questions_df = pd.DataFrame(questions_data)
print(f"Loaded {len(questions_df)} questions.")

# 2. Jobs Data
jobs_data = {
    'job_id': [1, 2, 3, 4, 5],
    'title': ["Data Scientist", "ML Engineer", "Data Analyst", "Research Scientist", "Data Engineer"],
    'company': ["TechCorp", "AIStartup", "FinTech Inc", "PharmaLabs", "BigData Co"],
    'description': [
        "Looking for a data scientist with strong ML skills to work on recommendation algorithms. Requires Python, SQL, Machine Learning, Statistics, A/B Testing.",
        "Develop and deploy machine learning models for computer vision applications using PyTorch, TensorFlow, MLOps, and Python.",
        "Analyze financial data to provide insights and create dashboards using SQL, Excel, Tableau, Statistics, and Financial Analysis skills.",
        "Apply advanced statistical methods to analyze clinical trial data using R, Statistics, Experimental Design, Causal Inference.",
        "Design and implement data pipelines for big data processing using Spark, Hadoop, SQL, Python, and Cloud Platforms (AWS/GCP)."
    ],
     'link': ["#job1", "#job2", "#job3", "#job4", "#job5"], # Using placeholder links
     'skills_required': [
         ["Python", "SQL", "Machine Learning", "Statistics", "A/B Testing"],
         ["PyTorch", "TensorFlow", "Computer Vision", "MLOps", "Python"],
         ["SQL", "Excel", "Tableau", "Statistics", "Financial Analysis"],
         ["R", "Statistics", "Experimental Design", "Causal Inference"],
         ["Spark", "Hadoop", "SQL", "Python", "AWS", "GCP"]
     ]
}
jobs_df = pd.DataFrame(jobs_data)
# Pre-compute Job Embeddings
jobs_df['embeddings'] = None # Initialize column
if embedding_model and 'description' in jobs_df.columns:
    try:
        jobs_df['embeddings'] = jobs_df['description'].fillna('').apply(lambda x: embedding_model.encode(x) if x else None)
        print(f"Job embeddings computed for {jobs_df['embeddings'].notna().sum()} jobs.")
    except Exception as e:
        print(f"Error computing job embeddings during setup: {e}")
else:
    print("Warning: Cannot compute job embeddings (model or description column missing).")
print(f"Loaded {len(jobs_df)} jobs.")


# 3. Study Recommendations Data
study_rec_data_list = []
temp_study_data = [
    {"category": "Machine Learning", "resources": [{"title": "ML Mastery Blog", "type": "Blog"}, {"title": "PRML Book", "type": "Book"}]},
    {"category": "SQL", "resources": [{"title": "SQLZoo Practice", "type": "Practice"}, {"title": "Leetcode DB", "type": "Practice"}]},
    {"category": "Data Preprocessing", "resources": [{"title": "Feature Eng Book", "type": "Book"}, {"title": "Kaggle Data Cleaning Course", "type": "Course"}]},
    {"category": "Experimental Design", "resources": [{"title": "Trustworthy Exp Book", "type": "Book"}, {"title": "Udacity A/B Course", "type": "Course"}]},
    {"category": "Statistics", "resources": [{"title": "StatQuest YouTube", "type": "Video"}, {"title": "Practical Stats Book", "type": "Book"}]}
]
for item in temp_study_data:
     category = item['category']
     rec_text = f"For {category}, check out: " + ", ".join([f"{res['title']} ({res['type']})" for res in item['resources']])
     # Add skill_tag based on category name (simple mapping for this example)
     study_rec_data_list.append({'skill_tag': category, 'recommendation_text': rec_text})
study_rec_df = pd.DataFrame(study_rec_data_list)
print(f"Loaded {len(study_rec_df)} study recommendations.")

# 4. Negotiation Scenarios Data
negotiation_data = {
    'scenario_id': [1, 2, 3, 4],
    'role_type': ['FAANG', 'Startup', 'FAANG', 'Startup'],
    'recruiter_statement': [
        "We're prepared to offer you a base salary of $120,000.",
        "Our standard offer for this role includes a base of $90,000 and 0.1% equity.",
        "Based on your experience, the salary band allows us to offer $135,000.",
        "We can offer $100,000 base salary, plus standard benefits."
    ],
    'good_user_counter_example': [
        "Thank you for the offer! Based on my research and competing opportunities, I was expecting a base closer to $135,000. Is there flexibility?",
        "I appreciate the offer and the equity component. Given the market rate, I'd be looking for a base around $105,000. Can we discuss the vesting schedule?",
        "That's a strong offer, thank you. Considering the cost of living and market data, I believe my value aligns more with the $145,000-$150,000 range.",
        "Thanks! Could you provide details on the bonus structure and typical total compensation? I'm aiming for a total package value around $120,000."
    ],
    'feedback_hint': [
        "User countered reasonably, referenced market/other offers.",
        "User acknowledged equity, provided target base, asked clarifying question.",
        "User expressed thanks, justified higher range.",
        "User asked clarifying questions about total compensation."
    ]
}
negotiation_scenarios_df = pd.DataFrame(negotiation_data)
print(f"Loaded {len(negotiation_scenarios_df)} negotiation scenarios.")

# --- Placeholder for Core Logic Import (will populate core_logic.py next) ---
# We need the functions before we define routes that use them.
# For now, define dummy functions or import later. Let's define dummies.
class CoreLogicPlaceholder:
    def generate_question(*args, **kwargs): return "Dummy Q", "Dummy Ideal", "Dummy Skills", 999, "Medium"
    def evaluate_answer(*args, **kwargs): return {'points': 5, 'similarity_score': 5.0, 'content_score': 5.0, 'clarity_score': 5.0, 'depth_score': 5.0, 'qualitative_feedback': 'Dummy feedback.'}
    def generate_follow_up(*args, **kwargs): return "Dummy follow-up?"
    def calculate_benchmark(*args, **kwargs): return 50
    def recommend_study_topics(*args, **kwargs): return ["Study Dummy Topics"]
    def extract_cv_skills(*args, **kwargs): return ["dummy_skill"]
    def recommend_jobs(*args, **kwargs): return [{"title": "Dummy Job", "company": "Dummy Co", "link": "#", "similarity_score": 0.5}]
    def simulate_negotiation(*args, **kwargs): return "Dummy negotiation dialogue.\n---\nFEEDBACK:\n1. Dummy feedback."
    def suggest_resume_tweaks(*args, **kwargs): return ["Dummy resume tweak."]

core_logic = CoreLogicPlaceholder() # Use placeholder for now
print("Using placeholder core logic functions.")
# --- End Placeholder ---


# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the main HTML page."""
    session.clear() # Start fresh session on page load
    print("Rendering index page, clearing session.")
    return render_template('index.html')

# --- Route for starting interview ---
@app.route('/start_interview', methods=['POST'])
def start_interview():
    print("Received /start_interview request")
    session.clear() # Ensure a clean state for new interview

    # Extract CV if provided
    cv_text = request.json.get('cv_text', '')
    extracted_skills = []
    cv_embedding_list = None # Store as list for JSON compatibility
    if cv_text and core_logic and hasattr(core_logic, 'extract_cv_skills'):
        extracted_skills = core_logic.extract_cv_skills(cv_text)
        if embedding_model and hasattr(core_logic, 'get_text_embedding'):
            cv_emb = core_logic.get_text_embedding(cv_text) # Use core_logic version if refactored
            if cv_emb is not None:
                cv_embedding_list = cv_emb.tolist() # Convert numpy array to list

    # Initialize session state
    session['interview_history'] = []
    session['asked_question_ids'] = []
    session['total_session_points'] = 0.0
    session['current_difficulty'] = "Medium"
    session['num_questions_total'] = 3 # Set session length
    session['cv_skills'] = extracted_skills # Store extracted skills
    session['cv_embedding'] = cv_embedding_list # Store embedding (as list)

    # Get the first question
    try:
        q_text, q_ideal, q_skills, q_id, q_difficulty = core_logic.generate_question(
            difficulty=session['current_difficulty'],
            excluded_ids=session['asked_question_ids']
        )
        if q_id < 0: # Check for error codes
             return jsonify({"error": "Could not generate first question."}), 500

        # Store current question details in session
        session['current_question'] = {
            'id': q_id, 'text': q_text, 'ideal': q_ideal,
            'skills': q_skills, 'difficulty': q_difficulty
        }
        session['asked_question_ids'].append(q_id)
        session.modified = True # Important when modifying mutable types in session

        return jsonify({
            "question_number": 1,
            "total_questions": session['num_questions_total'],
            "question_text": q_text,
            "category": q_skills, # Assuming skills map somewhat to category
            "difficulty": q_difficulty
        })
    except Exception as e:
        print(f"Error in /start_interview: {e}")
        return jsonify({"error": f"An error occurred: {e}"}), 500


# --- Add other routes here (/submit_answer, /simulate_negotiation, etc.) ---


# --- Run the App ---
if __name__ == '__main__':
    print("Starting Flask development server...")
    # Use host='0.0.0.0' for accessibility within network/Docker
    # Port 5000 is common for Flask dev
    app.run(debug=True, host='0.0.0.0', port=5000)