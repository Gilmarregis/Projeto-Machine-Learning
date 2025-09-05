# Implementar um algoritmo Machine learning para rodar no backend.
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=== ANÁLISE DE SOBREVIVÊNCIA - DATASET HABERMAN ===")
print("Atributos:")
print("1. Age: Idade do paciente na operação")
print("2. Year operation: Ano da operação (ano - 1900)")
print("3. Axillary nodes detected: Número de nódulos axilares detectados")
print("4. Survival status: Status de sobrevivência")
print("   1 = paciente sobreviveu 5 anos ou mais")
print("   2 = paciente morreu em menos de 5 anos")
print("\n" + "="*60 + "\n")

# Carregar dataset
url = "dados/haberman.csv"
names = ['Age', 'Year_operation', 'Axillary_nodes_detected', 'Survival_status']
dataset = pd.read_csv(url, names=names)

print(f"Dataset carregado com {len(dataset)} registros")
print("\nPrimeiras 5 linhas:")
print(dataset.head())

print("\nEstatísticas descritivas:")
print(dataset.describe())

print("\nDistribuição das classes:")
print(dataset['Survival_status'].value_counts())

# Preparar dados
array = dataset.values
X = array[:, :3]  # Features: Age, Year_operation, Axillary_nodes_detected
Y = array[:, 3]   # Target: Survival_status

# Divisão treino/validação
validation_size = 0.30
seed = 10
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, Y, test_size=validation_size, random_state=seed
)

print(f"\nDados de treino: {len(X_train)} amostras")
print(f"Dados de validação: {len(X_validation)} amostras")

# Configurações para validação cruzada
num_folds = 10
seed = 10
scoring = 'accuracy'

print("\n" + "="*60)
print("COMPARAÇÃO DE ALGORITMOS DE MACHINE LEARNING")
print("="*60)

# Lista de algoritmos para testar
algorithms = []
algorithms.append(('LR', LogisticRegression(random_state=seed, max_iter=1000)))
algorithms.append(('LDA', LinearDiscriminantAnalysis()))
algorithms.append(('KNN', KNeighborsClassifier()))
algorithms.append(('CART', DecisionTreeClassifier(random_state=seed)))
algorithms.append(('NB', GaussianNB()))
algorithms.append(('SVM', SVC(random_state=seed)))
algorithms.append(('NN', MLPClassifier(random_state=seed, max_iter=1000)))
algorithms.append(('RFC', RandomForestClassifier(random_state=seed)))

# Avaliar cada modelo
results = []
names = []
best_score = 0
best_model = None
best_name = ""

for name, algorithm in algorithms:
    kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    cv_results = cross_val_score(algorithm, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    
    mean_score = cv_results.mean()
    std_score = cv_results.std()
    
    print(f"{name:4}: {mean_score:.4f} (+/- {std_score:.4f})")
    
    # Identificar melhor modelo
    if mean_score > best_score:
        best_score = mean_score
        best_model = algorithm
        best_name = name

print(f"\n🏆 MELHOR MODELO: {best_name} com acurácia de {best_score:.4f}")

# Treinar o melhor modelo com todos os dados de treino
print(f"\nTreinando {best_name} com dados completos...")
best_model.fit(X_train, Y_train)

# Fazer predições no conjunto de validação
predictions = best_model.predict(X_validation)
accuracy = accuracy_score(Y_validation, predictions)

print(f"\n📊 RESULTADOS NO CONJUNTO DE VALIDAÇÃO:")
print(f"Acurácia: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\nMatriz de Confusão:")
print(confusion_matrix(Y_validation, predictions))
print("\nRelatório de Classificação:")
print(classification_report(Y_validation, predictions))

# Salvar o melhor modelo
model_filename = f'best_model_{best_name.lower()}.pkl'
joblib.dump(best_model, model_filename)
print(f"\n💾 Modelo salvo como '{model_filename}'")

# Também salvar informações do modelo
model_info = {
    'model_name': best_name,
    'accuracy': accuracy,
    'cv_score': best_score,
    'features': names,
    'classes': [1, 2],
    'class_names': ['Sobreviveu >= 5 anos', 'Morreu < 5 anos']
}

import json
with open('model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("📋 Informações do modelo salvas em 'model_info.json'")

print("\n" + "="*60)
print("TESTE COM DADOS FICTÍCIOS")
print("="*60)

# Criar dados fictícios para teste
np.random.seed(42)
n_samples = 15

# Gerar dados realistas baseados nas estatísticas do dataset original
fictitious_data = {
    'Age': np.random.randint(30, 70, n_samples),
    'Year_operation': np.random.randint(58, 70, n_samples),
    'Axillary_nodes_detected': np.random.randint(0, 30, n_samples)
}

df_fictitious = pd.DataFrame(fictitious_data)
print("\nDados fictícios gerados:")
print(df_fictitious)

# Salvar dados fictícios em CSV
df_fictitious.to_csv('dados_ficticios.csv', index=False)
print("\n💾 Dados fictícios salvos em 'dados_ficticios.csv'")

# Aplicar modelo nos dados fictícios
fictitious_predictions = best_model.predict(df_fictitious.values)
fictitious_probabilities = best_model.predict_proba(df_fictitious.values)

print("\n🔮 PREDIÇÕES PARA DADOS FICTÍCIOS:")
for i, (pred, prob) in enumerate(zip(fictitious_predictions, fictitious_probabilities)):
    status = "Sobreviveu >= 5 anos" if pred == 1 else "Morreu < 5 anos"
    confidence = max(prob) * 100
    print(f"Paciente {i+1:2d}: {status} (Confiança: {confidence:.1f}%)")

# Criar DataFrame com resultados
results_df = df_fictitious.copy()
results_df['Prediction'] = fictitious_predictions
results_df['Survival_Status'] = ['Sobreviveu >= 5 anos' if p == 1 else 'Morreu < 5 anos' 
                                for p in fictitious_predictions]
results_df['Confidence'] = [max(prob) * 100 for prob in fictitious_probabilities]

# Salvar resultados
results_df.to_csv('resultados_predicoes.csv', index=False)
print("\n💾 Resultados salvos em 'resultados_predicoes.csv'")

# Estatísticas das predições
survival_count = sum(fictitious_predictions == 1)
death_count = sum(fictitious_predictions == 2)

print(f"\n📈 RESUMO DAS PREDIÇÕES:")
print(f"Pacientes que sobreviverão >= 5 anos: {survival_count} ({survival_count/n_samples*100:.1f}%)")
print(f"Pacientes que morrerão < 5 anos: {death_count} ({death_count/n_samples*100:.1f}%)")

# Visualização simples
plt.figure(figsize=(12, 8))

# Subplot 1: Distribuição das predições
plt.subplot(2, 2, 1)
plt.bar(['Sobreviveu >= 5 anos', 'Morreu < 5 anos'], [survival_count, death_count])
plt.title('Distribuição das Predições')
plt.ylabel('Número de Pacientes')

# Subplot 2: Idade vs Predição
plt.subplot(2, 2, 2)
colors = ['green' if p == 1 else 'red' for p in fictitious_predictions]
plt.scatter(df_fictitious['Age'], fictitious_predictions, c=colors, alpha=0.7)
plt.xlabel('Idade')
plt.ylabel('Predição (1=Sobreviveu, 2=Morreu)')
plt.title('Idade vs Predição de Sobrevivência')

# Subplot 3: Nódulos vs Predição
plt.subplot(2, 2, 3)
plt.scatter(df_fictitious['Axillary_nodes_detected'], fictitious_predictions, c=colors, alpha=0.7)
plt.xlabel('Nódulos Axilares Detectados')
plt.ylabel('Predição (1=Sobreviveu, 2=Morreu)')
plt.title('Nódulos vs Predição de Sobrevivência')

# Subplot 4: Confiança das predições
plt.subplot(2, 2, 4)
confidences = [max(prob) * 100 for prob in fictitious_probabilities]
plt.hist(confidences, bins=5, alpha=0.7, color='blue')
plt.xlabel('Confiança da Predição (%)')
plt.ylabel('Frequência')
plt.title('Distribuição da Confiança das Predições')

plt.tight_layout()
plt.savefig('analise_predicoes.png', dpi=300, bbox_inches='tight')
print("\n📊 Gráficos salvos em 'analise_predicoes.png'")

print("\n" + "="*60)
print("✅ ANÁLISE COMPLETA FINALIZADA!")
print("="*60)
print("\nArquivos gerados:")
print(f"• {model_filename} - Modelo treinado")
print("• model_info.json - Informações do modelo")
print("• dados_ficticios.csv - Dataset fictício")
print("• resultados_predicoes.csv - Predições com confiança")
print("• analise_predicoes.png - Visualizações")
print("\n🎯 O modelo está pronto para uso em produção!")

