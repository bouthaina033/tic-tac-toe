import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Patterns de victoire
WIN_PATTERNS = [
    (0,1,2),(3,4,5),(6,7,8),
    (0,3,6),(1,4,7),(2,5,8),
    (0,4,8),(2,4,6)
]

# 0 = X (humain), 1 = O (IA), -1 = vide

def check_winner(board):
    for a, b, c in WIN_PATTERNS:
        if board[a] != -1 and board[a] == board[b] == board[c]:
            return board[a]
    return None


def minimax(board, player):
    winner = check_winner(board)
    if winner is not None:
        return (1, None) if winner == 1 else (-1, None)
    if all(cell != -1 for cell in board):
        return (0, None)

    best_score = -2 if player == 1 else 2
    best_move = None
    for i in range(9):
        if board[i] == -1:
            board[i] = player
            score, _ = minimax(board, 1-player)
            board[i] = -1
            if (player == 1 and score > best_score) or (player == 0 and score < best_score):
                best_score, best_move = score, i
    return best_score, best_move


def generate_dataset():
    rows = []
    def recurse(board, turn):
        # si terminal, stop
        if check_winner(board) is not None or all(c != -1 for c in board):
            return
        # si tour IA, enregistre état + meilleur coup
        if turn == 1:
            _, move = minimax(board[:], 1)
            rows.append(board[:] + [move])
        # explore coups suivants
        for i in range(9):
            if board[i] == -1:
                board[i] = turn
                recurse(board, 1-turn)
                board[i] = -1

    recurse([-1]*9, 0)
    df = pd.DataFrame(rows, columns=[f'c{i}' for i in range(9)] + ['best_move'])
    df.to_csv('tictactoe_moves.csv', index=False)
    print(f"Dataset généré: {len(df)} états")
    return df


if __name__ == '__main__':
    # 1) Génération
    df = generate_dataset()
    X = df.drop('best_move', axis=1)
    y = df['best_move']

    # 2) Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3) Standardisation
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # 4) Définition des modèles
    models = {
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3),
        'Support Vector Machine': SVC(probability=False),
        'Decision Tree': DecisionTreeClassifier(max_depth=5),
        'Random Forest': RandomForestClassifier(n_estimators=100)
    }


    # 5) Entraînement et sauvegarde
    import os
    os.makedirs('../models', exist_ok=True)
    joblib.dump(scaler, '../models/scaler_move.pkl')

    for name, model in models.items():
        model.fit(X_train_s, y_train)
        acc = model.score(X_test_s, y_test)
        print(f"{name} - Test accuracy: {acc:.3f}")
        filename = f"../models/model_move_{name.replace(' ', '_')}.pkl"
        joblib.dump(model, filename)
    print("Entraînement terminé et modèles sauvegardés dans /models")