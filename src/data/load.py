import pandas as pd
import joblib
from pathlib import Path
from typing import Any

# Linha 6: falta import Config
from src.utils.config import Config  # ← ADICIONAR ESTA LINHA

class DataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> None:
        """Salva dados processados"""
        output_path = Path(self.config.processed_data_path) / filename
        df.to_csv(output_path, index=False)
        self.logger.info(f"Dados salvos em: {output_path}")
    
    def save_model(self, model: Any, model_name: str, metadata: dict) -> None:
        """Salva modelo com metadados"""
        model_path = Path(self.config.models_path) / f"{model_name}.pkl"
        metadata_path = Path(self.config.models_path) / f"{model_name}_metadata.json"
        
        joblib.dump(model, model_path)
        
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Modelo salvo: {model_path}")