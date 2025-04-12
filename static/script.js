// ==========================================
// static/script.js - Frontend Logic for MVP
// ==========================================
console.log("JobSage MVP script loaded.");

// --- DOM Elements ---
const startBtn = document.getElementById('start-btn');
const questionArea = document.getElementById('question-area');
const questionHeader = document.getElementById('question-header');
const questionText = document.getElementById('question-text');
const questionDetails = document.getElementById('question-details');
const answerInput = document.getElementById('answer-input');
const submitBtn = document.getElementById('submit-btn');
const evaluationArea = document.getElementById('evaluation-area');
const pointsDisplay = document.getElementById('points-display'); // Placeholder, maybe add later
const feedbackDisplay = document.getElementById('feedback-display');
const followUpDisplay = document.getElementById('follow-up-display');
const progressDisplay = document.getElementById('progress-display');
const interviewCompleteMessage = document.getElementById('interview-complete-message');
const finalPointsDisplay = document.getElementById('final-points-display');
const finalBenchmarkDisplay = document.getElementById('final-benchmark-display');
const cvInput = document.getElementById('cv-input');

const negotiateBtn = document.getElementById('negotiate-btn');
const jobTitleInput = document.getElementById('job-title-input');
const yearsExpInput = document.getElementById('years-exp-input');
const currentSalaryInput = document.getElementById('current-salary-input');
const negotiationOutput = document.getElementById('negotiation-output');

// --- State (Simple Frontend State) ---
let currentQuestionNumber = 0;
let totalQuestions = 0;

// --- Event Listeners ---
if (startBtn) startBtn.addEventListener('click', handleStartInterview);
if (submitBtn) submitBtn.addEventListener('click', handleSubmitAnswer);
if (negotiateBtn) negotiateBtn.addEventListener('click', handleNegotiation);

// --- API Call Functions ---
async function postData(url = '', data = {}) {
    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });
        if (!response.ok) {
            // Try to parse error message from backend
            const errorData = await response.json().catch(() => ({ error: `HTTP error! status: ${response.status}` }));
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }
        return await response.json(); // Parses JSON response body
    } catch (error) {
        console.error('Error during fetch:', error);
        throw error; // Re-throw the error to be caught by the caller
    }
}

// --- Event Handlers ---
async function handleStartInterview() {
    console.log("Start Interview Clicked");
    startBtn.disabled = true;
    startBtn.innerText = "Starting...";
    progressDisplay.innerText = "Initializing interview...";
    questionArea.classList.add('hidden');
    evaluationArea.classList.add('hidden');
    interviewCompleteMessage.classList.add('hidden');
    feedbackDisplay.innerText = ""; // Clear feedback
    followUpDisplay.innerText = ""; // Clear follow-up


    const cvText = cvInput.value;

    try {
        const data = await postData('/start_interview', { cv_text: cvText });

        if (data.error) {
            alert(`Error starting interview: ${data.error}`);
            progressDisplay.innerText = "Error starting.";
            startBtn.disabled = false;
            startBtn.innerText = "Start New Interview Session";
            return;
        }

        // Update state
        currentQuestionNumber = data.question_number;
        totalQuestions = data.total_questions;

        // Update UI
        questionArea.classList.remove('hidden');
        questionHeader.innerText = `Question ${currentQuestionNumber}/${totalQuestions}`;
        questionText.innerText = data.question_text;
        questionDetails.innerText = `Category: ${data.category} | Difficulty: ${data.difficulty}`;
        answerInput.value = "";
        answerInput.disabled = false;
        submitBtn.disabled = false;
        progressDisplay.innerText = `Question ${currentQuestionNumber} of ${totalQuestions}`;
        startBtn.innerText = "Restart Interview Session"; // Change button text

    } catch (error) {
        progressDisplay.innerText = `Error: ${error.message}`;
        alert(`Failed to start interview: ${error.message}`);
        startBtn.innerText = "Start New Interview Session"; // Reset button text
    } finally {
        startBtn.disabled = false; // Re-enable start button
    }
}

async function handleSubmitAnswer() {
    console.log("Submit Answer Clicked");
    const userAnswer = answerInput.value;
    if (!userAnswer.trim()) {
        alert("Please enter an answer.");
        return;
    }

    submitBtn.disabled = true; // Prevent double clicks
    submitBtn.innerText = "Evaluating...";
    feedbackDisplay.innerText = "Evaluating answer..."; // Loading state
    evaluationArea.classList.remove('hidden'); // Show evaluation area
    followUpDisplay.innerText = ""; // Clear previous follow-up


    try {
        const data = await postData('/submit_answer', { answer: userAnswer });

        if (data.error) {
            alert(`Error submitting answer: ${data.error}`);
            feedbackDisplay.innerText = `Error: ${data.error}`;
            submitBtn.innerText = "Submit Answer";
            submitBtn.disabled = false; // Re-enable on error
            return;
        }

        // Display evaluation
        pointsDisplay.innerText = `Points this Q: ${data.evaluation?.points ?? 'N/A'} / ${MAX_POINTS_PER_QUESTION}`; // Update points (might need MAX_POINTS_PER_QUESTION from backend or define here)
        feedbackDisplay.innerText = data.evaluation?.qualitative_feedback ?? 'No feedback available.';
        followUpDisplay.innerText = data.follow_up ? `Follow-up: ${data.follow_up}` : "";

        // Handle interview completion or next question
        if (data.interview_complete) {
            questionArea.classList.add('hidden'); // Hide question input
            interviewCompleteMessage.classList.remove('hidden'); // Show completion message
            finalPointsDisplay.innerText = `Final Total Points: ${data.current_total_points?.toFixed(1) ?? 'N/A'}`;
            // TODO: Add call to /get_results here to populate benchmark, recs etc.
            // finalBenchmarkDisplay.innerText = `Benchmark: Top X%`; // Update after /get_results call
            progressDisplay.innerText = `Finished ${totalQuestions} questions.`;
            answerInput.disabled = true;
            submitBtn.disabled = true;
            submitBtn.innerText = "Interview Complete";

        } else if (data.next_question) {
            // Update for next question
            currentQuestionNumber = data.next_question.question_number;
            questionHeader.innerText = `Question ${currentQuestionNumber}/${totalQuestions}`;
            questionText.innerText = data.next_question.question_text;
            questionDetails.innerText = `Category: ${data.next_question.category} | Difficulty: ${data.next_question.difficulty}`;
            answerInput.value = ""; // Clear input
            answerInput.disabled = false;
            submitBtn.disabled = false; // Re-enable for next question
            submitBtn.innerText = "Submit Answer";
            progressDisplay.innerText = `Question ${currentQuestionNumber} of ${totalQuestions}`;
        } else {
            // Should not happen if interview_complete is false, but handle defensively
            console.error("Unexpected state: Interview not complete, but no next question.");
            feedbackDisplay.innerText += "\nError: Could not load next question.";
            answerInput.disabled = true;
            submitBtn.disabled = true;
        }

    } catch (error) {
        feedbackDisplay.innerText = `Error: ${error.message}`;
        alert(`Failed to submit answer: ${error.message}`);
        submitBtn.disabled = false; // Re-enable on error
        submitBtn.innerText = "Submit Answer";
    }
}

async function handleNegotiation() {
    console.log("Negotiate Clicked");
    const jobTitle = jobTitleInput.value;
    const yearsExp = yearsExpInput.value;
    const currentSalary = currentSalaryInput.value;

    if (!currentSalary || !jobTitle || !yearsExp) {
         alert("Please fill in all negotiation fields.");
         return;
    }
    negotiateBtn.disabled = true;
    negotiateBtn.innerText = "Simulating...";
    negotiationOutput.innerText = "Simulating negotiation with AI... Please wait.";

    try {
        const data = await postData('/simulate_negotiation', {
            job_title: jobTitle,
            years_experience: parseInt(yearsExp, 10), // Ensure number
            current_salary: parseInt(currentSalary, 10) // Ensure number
        });

        if (data.error) {
            alert(`Error simulating negotiation: ${data.error}`);
            negotiationOutput.innerText = `Error: ${data.error}`;
        } else {
            negotiationOutput.innerText = data.simulation_text ?? "Simulation generated no text.";
        }

    } catch (error) {
        negotiationOutput.innerText = `Error: ${error.message}`;
        alert(`Failed to simulate negotiation: ${error.message}`);
    } finally {
        negotiateBtn.disabled = false;
        negotiateBtn.innerText = "Simulate Negotiation";
    }
}

// --- Initial State Setup ---
function initializeUI() {
    console.log("Initializing UI state.");
    submitBtn.disabled = true;
    answerInput.disabled = true;
    questionArea.classList.add('hidden');
    evaluationArea.classList.add('hidden');
    interviewCompleteMessage.classList.add('hidden');
}

// Run initialization when the script loads
initializeUI();