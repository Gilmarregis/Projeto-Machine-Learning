from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Carregar o modelo treinado
loaded_model = joblib.load('logistic_regression_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    idade = data['idade']
    renda_anual = data['renda_anual']

    # Fazer a predição
    prediction = loaded_model.predict([[idade, renda_anual]])
    prediction_proba = loaded_model.predict_proba([[idade, renda_anual]])[0]

    result = {
        'comprou_produto': int(prediction[0]),
        'probabilidade_nao_comprou': prediction_proba[0],
        'probabilidade_comprou': prediction_proba[1]
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
