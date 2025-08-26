#Implementar um algoritmo Machine learning para rodar no backend.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib


# 1. Coleta de Dados (Exemplo: dados fictícios de clientes)
data = {
    'idade': [25, 30, 35, 40, 45, 50, 28, 33, 38, 42],
    'renda_anual': [30000, 40000, 50000, 60000, 70000, 80000, 35000, 45000, 55000, 65000],
    'comprou_produto': [0, 0, 1, 0, 1, 1, 0, 1, 0, 1]  # 0: Não comprou, 1: Comprou
}
df = pd.DataFrame(data)

# 2. Pré-processamento de Dados (se necessário, neste caso, os dados já estão limpos)
X = df[['idade', 'renda_anual']]
y = df['comprou_produto']

# 3. Divisão dos Dados em Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Escolha e Treinamento do Modelo (Exemplo: Regressão Logística)
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Avaliação do Modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy:.2f}")

# 6. Salvar o Modelo Treinado para uso no Backend
model_filename = 'logistic_regression_model.pkl'
joblib.dump(model, model_filename)
print(f"Modelo salvo como '{model_filename}'")

