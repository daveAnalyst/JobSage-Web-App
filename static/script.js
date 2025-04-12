// --- DOM Elements ---
const startBtn = document.getElementById('start-btn');
const questionArea = document.getElementById('question-area');
const questionHeader = document.getElementById('question-header');
const questionText = document.getElementById('question-text');
const questionDetails = document.getElementById('question-details');
const answerInput = document.getElementById('answer-input');
const submitBtn = document.getElementById('submit-btn');
const evaluationArea = document.getElementById('evaluation-area');
const pointsDisplay = document.getElementById('points-display');
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


// --- Event Listeners ---
startBtn.addEventListener('click', handleStartInterview);
submitBtn.addEventListener('click', handleSubmitAnswer);
negotiateBtn.addEventListener('click', handleNegotiation);

// --- Functions ---
async function handleStartInterview() {
    console.log("Start Interview Clicked");
    const cvText = cvInput.value;
    // TODO: Send POST request to Flask '/start_interview'
    // TODO: Handle response, display first question, enable submit button
    alert("Start Interview logic not fully implemented yet."); // Placeholder
    // Example structure after getting response:
    // questionArea.classList.remove('hidden');
    // evaluationArea.classList.add('hidden');
    // interviewCompleteMessage.classList.add('hidden');
    // questionHeader.innerText = `Question ${response.question_number}/${response.total_questions}`;
    // questionText.innerText = response.question_text;
    // questionDetails.innerText = `Category: ${response.category} | Difficulty: ${response.difficulty}`;
    // answerInput.value = "";
    // answerInput.disabled = false;
    // submitBtn.disabled = false;
    // progressDisplay.innerText = `Question ${response.question_number} of ${response.total_questions}`;
}

async function handleSubmitAnswer() {
    console.log("Submit Answer Clicked");
    const userAnswer = answerInput.value;
    if (!userAnswer.trim()) {
        alert("Please enter an answer.");
        return;
    }
    submitBtn.disabled = true; // Prevent double clicks
    feedbackDisplay.innerText = "Evaluating answer..."; // Loading state
    // TODO: Send POST request to Flask '/submit_answer' with userAnswer
    // TODO: Handle response (evaluation, follow-up, next question or end)
    // TODO: Update UI elements (feedback, points, follow-up, progress, next question)
    // TODO: Re-enable submit button if not end of interview
    alert("Submit Answer logic not fully implemented yet."); // Placeholder
     // Example structure after getting response:
     // evaluationArea.classList.remove('hidden');
     // pointsDisplay.innerText = `Points this Q: ${response.evaluation.points}`;
     // feedbackDisplay.innerText = response.evaluation.qualitative_feedback;
     // followUpDisplay.innerText = response.follow_up ? `Follow-up: ${response.follow_up}` : "";
     // if (response.next_question) {
     //    // Update question display for next question
     //    submitBtn.disabled = false;
     //    answerInput.value = "";
     // } else {
     //    // Handle end of interview
     //    interviewCompleteMessage.classList.remove('hidden');
     //    // ... display final stats ...
     // }
}

async function handleNegotiation() {
    console.log("Negotiate Clicked");
    const jobTitle = jobTitleInput.value;
    const yearsExp = yearsExpInput.value;
    const currentSalary = currentSalaryInput.value;

    if (!currentSalary) {
         alert("Please enter current salary.");
         return;
    }
    negotiateBtn.disabled = true;
    negotiationOutput.innerText = "Simulating negotiation..."; // Loading state

    // TODO: Send POST request to Flask '/simulate_negotiation' with inputs
    // TODO: Handle response and display in negotiationOutput
    alert("Negotiation logic not fully implemented yet."); // Placeholder
    negotiationOutput.innerText = "*Negotiation simulation & feedback will appear here...*"; // Reset placeholder
    negotiateBtn.disabled = false; // Re-enable after placeholder alert

}

// --- Initial State ---
// Disable submit button initially
submitBtn.disabled = true;
answerInput.disabled = true; // Disable answer input until interview starts

console.log("JobSage MVP script loaded.");