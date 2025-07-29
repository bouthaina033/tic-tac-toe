let currentPlayer = 'X';
let gameBoard = [-1, -1, -1, -1, -1, -1, -1, -1, -1];
let gameActive = true;

function createBoard() {
    const board = document.getElementById('board');
    board.innerHTML = '';

    for (let i = 0; i < 9; i++) {
        const cell = document.createElement('div');
        cell.className = 'cell';
        cell.setAttribute('data-index', i);
        cell.addEventListener('click', () => handleCellClick(i));
        board.appendChild(cell);
    }
}

async function handleCellClick(index) {
    if (!gameActive || gameBoard[index] !== -1 || currentPlayer === 'O') return;

    gameBoard[index] = 0; // Human player (X) = 0
    updateBoard();

    if (checkWinner()) return;

    currentPlayer = 'O';
    await aiMove();
}

async function aiMove() {
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            board: gameBoard,
            model: document.getElementById('modelSelector').value
        })
    });

    const data = await response.json();
    const move = data.move;

    if (gameBoard[move] === -1) {
        gameBoard[move] = 1; // AI (O) = 1
        updateBoard();
        checkWinner();
    }

    currentPlayer = 'X';
}

function updateBoard() {
    document.querySelectorAll('.cell').forEach((cell, index) => {
        cell.textContent = gameBoard[index] === 0 ? 'X' : gameBoard[index] === 1 ? 'O' : '';
    });
}

function checkWinner() {
    // Implement win/draw logic here
    return false;
}

function resetGame() {
    gameBoard = [-1, -1, -1, -1, -1, -1, -1, -1, -1];
    gameActive = true;
    currentPlayer = 'X';
    document.getElementById('status').textContent = '';
    createBoard();
}

// Initialize game
createBoard();