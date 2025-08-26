# Machine Learning Backend

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.x](https://img.shields.io/badge/python-3.x-brightgreen.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0-orange.svg)](https://scikit-learn.org/)

Um projeto de Machine Learning para predição de compra de produtos baseado em dados de clientes, implementado para uso em backend.

## 📋 Descrição

Este projeto implementa um algoritmo de Machine Learning usando Regressão Logística para prever se um cliente irá comprar um produto baseado em sua idade e renda anual. O modelo é treinado e salvo para ser utilizado em uma API backend.

## 🚀 Funcionalidades

- **Coleta e processamento de dados**: Utiliza dados fictícios de clientes
- **Treinamento de modelo**: Implementa Regressão Logística para classificação binária
- **Avaliação de performance**: Calcula a acurácia do modelo
- **Persistência do modelo**: Salva o modelo treinado para uso posterior
- **API Backend**: Implementação com Flask em `api_ML.py`

## 🛠️ Tecnologias Utilizadas

- **Python 3.x**
- **pandas**: Manipulação de dados
- **scikit-learn**: Algoritmos de Machine Learning
- **joblib**: Serialização do modelo
- **Flask**: Framework web (para API)

## Instalação

1. Certifique-se de ter o Python instalado (versão 3.8 ou superior recomendada).
2. Clone o repositório:
   ```bash
   git clone https://github.com/Gilmarregis/Projeto-Machine-Learning.git
   cd "Projeto-Machine-Learning"
   ```
3. Instale as dependências do projeto executando:
   ```bash
   pip install -r requirements.txt
   ```

## 🔧 Como Usar

### Executar o Treinamento do Modelo

```bash
python ML_model.py
```

Este comando irá:
- Carregar os dados de exemplo
- Treinar o modelo de Regressão Logística
- Avaliar a acurácia
- Salvar o modelo como `logistic_regression_model.pkl`

### Executar a API

```bash
python api_ML.py
```

A API estará disponível em http://localhost:5000/predict. Envie uma requisição POST com dados como:

```json
{
    "idade": 35,
    "renda_anual": 50000
}
```

#### Exemplo de Resposta:

```json
{
    "comprou_produto": 1,
    "probabilidade_nao_comprou": 0.25,
    "probabilidade_comprou": 0.75
}
```

## 📊 Estrutura dos Dados

O modelo utiliza as seguintes features:

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `idade` | int | Idade do cliente |
| `renda_anual` | int | Renda anual do cliente em reais |
| `comprou_produto` | int | Target: 0 (não comprou) ou 1 (comprou) |

## 📈 Performance do Modelo

O modelo atual utiliza um dataset pequeno de exemplo. Para melhor performance:

- Utilize um dataset maior e mais diversificado
- Considere feature engineering adicional
- Teste outros algoritmos (Random Forest, SVM, etc.)
- Implemente validação cruzada

## 🔄 Fluxo do Projeto

1. **Coleta de Dados**: Carregamento dos dados de clientes
2. **Pré-processamento**: Preparação dos dados (já limpos no exemplo)
3. **Divisão**: Split em dados de treino e teste (80/20)
4. **Treinamento**: Fit do modelo de Regressão Logística
5. **Avaliação**: Cálculo da acurácia
6. **Persistência**: Salvamento do modelo treinado

## 📁 Estrutura do Projeto

```
Machine Learning/
├── Machine Learning Backend.py    # Código principal
├── README.md                     # Este arquivo
└── logistic_regression_model.pkl # Modelo treinado (gerado após execução)
```

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 📞 Contato

Para dúvidas ou sugestões, entre em contato através dos issues do GitHub.

---

⭐ Se este projeto foi útil para você, considere dar uma estrela!
