import time
from functools import wraps
from prometheus_client import Counter, Histogram, Gauge

# Métricas Prometheus
PREDICTION_COUNTER = Counter('ml_predictions_total', 'Total predictions', ['model', 'status'])
PREDICTION_LATENCY = Histogram('ml_prediction_duration_seconds', 'Prediction latency')
MODEL_ACCURACY = Gauge('ml_model_accuracy', 'Current model accuracy', ['model'])

def monitor_predictions(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            PREDICTION_COUNTER.labels(model='haberman', status='success').inc()
            return result
        except Exception as e:
            PREDICTION_COUNTER.labels(model='haberman', status='error').inc()
            raise
        finally:
            PREDICTION_LATENCY.observe(time.time() - start_time)
    return wrapper