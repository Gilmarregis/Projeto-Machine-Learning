import logging
import json
from datetime import datetime
from typing import Dict, Any

class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_model_prediction(self, model_name: str, input_data: Dict, 
                           prediction: Any, confidence: float):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'model_prediction',
            'model_name': model_name,
            'input_features': input_data,
            'prediction': prediction,
            'confidence': confidence
        }
        self.logger.info(json.dumps(log_data))
    
    def log_model_training(self, model_name: str, metrics: Dict[str, float]):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': 'model_training',
            'model_name': model_name,
            'metrics': metrics
        }
        self.logger.info(json.dumps(log_data))