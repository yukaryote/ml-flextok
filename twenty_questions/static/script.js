// API base URL
const API_URL = 'http://127.0.0.1:5000/api';

// Game state
let currentQuestion = 0;
let maxQuestions = 20;
let isGameActive = false;

// DOM elements
const elements = {
    progress: document.getElementById('progress'),
    progressFill: document.getElementById('progress-fill'),
    status: document.getElementById('status'),
    loading: document.getElementById('loading'),
    gameArea: document.getElementById('game-area'),
    finalResult: document.getElementById('final-result'),
    startBtn: document.getElementById('start-btn'),
    restartBtn: document.getElementById('restart-btn'),
    imgA: document.getElementById('img-a'),
    imgB: document.getElementById('img-b'),
    finalImg: document.getElementById('final-img'),
    chosenHistory: document.getElementById('chosen-history'),
    rejectedHistory: document.getElementById('rejected-history')
};

// Show/hide elements
function show(...els) {
    els.forEach(el => el.classList.remove('hidden'));
}

function hide(...els) {
    els.forEach(el => el.classList.add('hidden'));
}

// Update progress
function updateProgress() {
    const percentage = (currentQuestion / maxQuestions) * 100;
    elements.progressFill.style.width = `${percentage}%`;

    const progressText = document.querySelector('.progress-text');
    if (currentQuestion === 0) {
        progressText.textContent = 'Ready to start';
    } else if (currentQuestion >= maxQuestions) {
        progressText.textContent = 'Complete!';
    } else {
        progressText.textContent = `Question ${currentQuestion} of ${maxQuestions}`;
    }
}

// Show loading state
function showLoading(message = 'Generating images...') {
    elements.loading.querySelector('p').textContent = message;
    show(elements.loading);
    hide(elements.gameArea, elements.finalResult);
}

// Update status message
function updateStatus(message) {
    elements.status.textContent = message;
}

// Start new game
async function startGame() {
    try {
        showLoading('Initializing game...');
        hide(elements.startBtn, elements.restartBtn, elements.finalResult);
        updateStatus('Starting new game...');

        // Clear history
        elements.chosenHistory.innerHTML = '';
        elements.rejectedHistory.innerHTML = '';
        currentQuestion = 0;
        updateProgress();

        const response = await fetch(`${API_URL}/init`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        if (!response.ok) {
            throw new Error('Failed to start game');
        }

        const data = await response.json();

        currentQuestion = data.question;
        maxQuestions = data.max_questions;
        isGameActive = true;

        // Display options
        elements.imgA.src = data.option_a;
        elements.imgB.src = data.option_b;

        hide(elements.loading);
        show(elements.gameArea);
        updateProgress();
        updateStatus('Which option better matches your imagined face?');

    } catch (error) {
        console.error('Error starting game:', error);
        // Check if it's a network error
        let errorMsg = error.message;
        if (error.message === 'Load failed' || error.message.includes('fetch')) {
            errorMsg = 'Cannot connect to server. Make sure Flask backend is running on http://127.0.0.1:5000';
        }
        updateStatus(`Error: ${errorMsg}`);
        hide(elements.loading);
        show(elements.startBtn);
    }
}

// Make choice
async function chooseOption(choice) {
    if (!isGameActive) return;

    try {
        showLoading();
        updateStatus('Processing choice...');

        const response = await fetch(`${API_URL}/choose`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ choice })
        });

        if (!response.ok) {
            throw new Error('Failed to process choice');
        }

        const data = await response.json();

        // Add to history
        addToHistory(data.chosen, data.rejected);

        if (data.status === 'complete') {
            // Game complete
            isGameActive = false;
            currentQuestion = data.question;
            updateProgress();

            elements.finalImg.src = data.final_image;

            hide(elements.loading, elements.gameArea);
            show(elements.finalResult, elements.restartBtn);
            updateStatus('Game complete! Here is your face.');

        } else {
            // Continue to next question
            currentQuestion = data.question;
            updateProgress();

            elements.imgA.src = data.option_a;
            elements.imgB.src = data.option_b;

            hide(elements.loading);
            show(elements.gameArea);
            updateStatus('Which option better matches your imagined face?');
        }

    } catch (error) {
        console.error('Error making choice:', error);
        updateStatus('Error processing choice. Please try again.');
        hide(elements.loading);
        show(elements.gameArea);
    }
}

// Add images to history
function addToHistory(chosenBase64, rejectedBase64) {
    // Add chosen image
    const chosenImg = document.createElement('img');
    chosenImg.src = chosenBase64;
    chosenImg.alt = 'Chosen';
    elements.chosenHistory.appendChild(chosenImg);

    // Add rejected image
    const rejectedImg = document.createElement('img');
    rejectedImg.src = rejectedBase64;
    rejectedImg.alt = 'Rejected';
    elements.rejectedHistory.appendChild(rejectedImg);

    // Auto-scroll to latest
    elements.chosenHistory.scrollLeft = elements.chosenHistory.scrollWidth;
    elements.rejectedHistory.scrollLeft = elements.rejectedHistory.scrollWidth;
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    updateProgress();
});
