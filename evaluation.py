import pandas as pd
import time
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Charger les données et modèles (code inchangé)
df = pd.read_csv('C:/Users/aya/PycharmProjects/ML_TP/ML_Projet_Correct/src/tictactoe_moves.csv')
X = df.drop('best_move', axis=1)
y = df['best_move']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = joblib.load('C:/Users/aya/PycharmProjects/ML_TP/ML_Projet_Correct/models/scaler_move.pkl')
X_test_s = scaler.transform(X_test)

models = {
    'K-Nearest Neighbors': 'model_move_K-Nearest_Neighbors.pkl',
    'Support Vector Machine': 'model_move_Support_Vector_Machine.pkl',
    'Decision Tree': 'model_move_Decision_Tree.pkl',
    'Random Forest': 'model_move_Random_Forest.pkl'
}

# Nouvelle structure de stockage des résultats
metrics = {
    'Modèle': [],
    'Précision': [],
    'Rappel': [],
    'F1-score': [],
    'Exactitude': [],
    'Temps de prédiction (ms)': []
}

# Évaluation (code similaire)
for model_name, model_file in models.items():
    model = joblib.load(f'C:/Users/aya/PycharmProjects/ML_TP/ML_Projet_Correct/models/{model_file}')

    start_time = time.time()
    y_pred = model.predict(X_test_s)
    prediction_time = time.time() - start_time

    metrics['Modèle'].append(model_name)
    metrics['Précision'].append(precision_score(y_test, y_pred, average='weighted'))
    metrics['Rappel'].append(recall_score(y_test, y_pred, average='weighted'))
    metrics['F1-score'].append(f1_score(y_test, y_pred, average='weighted'))
    metrics['Exactitude'].append(accuracy_score(y_test, y_pred))
    metrics['Temps de prédiction (ms)'].append(prediction_time * 1000)

results_df = pd.DataFrame(metrics)

# 1. Graphique en barres pour les métriques de performance
plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")

ax = sns.barplot(
    x='Modèle',
    y='value',
    hue='variable',
    data=pd.melt(results_df, id_vars=['Modèle'], value_vars=['Précision', 'Rappel', 'F1-score', 'Exactitude'])
)

plt.title('Comparaison des métriques de performance', fontsize=14)
plt.ylabel('Score')
plt.ylim(0.4, 1.0)
plt.legend(loc='lower right')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('performance_metrics.png', dpi=300)
plt.close()

# 2. Graphique en ligne pour le temps de prédiction
plt.figure(figsize=(10, 5))
sns.lineplot(
    x='Modèle',
    y='Temps de prédiction (ms)',
    data=results_df,
    marker='o',
    linewidth=2.5,
    markersize=10
)

plt.title('Temps de prédiction par modèle', fontsize=14)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('prediction_time.png', dpi=300)
plt.close()

print("Visualisations générées : performance_metrics.png et prediction_time.png")

# Optionnel : Sauvegarder les résultats
results_df.to_csv('C:/Users/aya/PycharmProjects/ML_TP/ML_Projet_Correct/models/model_evaluation_results.csv', index=False)