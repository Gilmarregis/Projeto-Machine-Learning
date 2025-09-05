"""
Módulo de configuração do projeto ML.

Este módulo contém a classe Config que centraliza todas as configurações
do projeto, incluindo caminhos de dados, modelos e parâmetros.
"""

import os
from pathlib import Path
from typing import Optional

class Config:
    """
    Classe de configuração centralizada para o projeto ML.
    
    Attributes:
        base_path (Path): Caminho base do projeto
        data_path (Path): Caminho para dados brutos
        processed_data_path (Path): Caminho para dados processados
        models_path (Path): Caminho para modelos salvos
        logs_path (Path): Caminho para logs
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Inicializa a configuração do projeto.
        
        Args:
            base_path (Optional[str]): Caminho base customizado
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.data_path = self.base_path / "dados"
        self.processed_data_path = self.base_path / "data" / "processed"
        self.models_path = self.base_path / "models" / "trained"
        self.logs_path = self.base_path / "logs"
        
        # Criar diretórios se não existirem
        self._create_directories()
    
    def _create_directories(self) -> None:
        """
        Cria os diretórios necessários se não existirem.
        """
        directories = [
            self.processed_data_path,
            self.models_path,
            self.logs_path
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def haberman_dataset_path(self) -> Path:
        """
        Retorna o caminho para o dataset Haberman.
        
        Returns:
            Path: Caminho para haberman.csv
        """
        return self.data_path / "haberman.csv"
    
    @property
    def mlflow_tracking_uri(self) -> str:
        """
        Retorna a URI de tracking do MLflow.
        
        Returns:
            str: URI do MLflow
        """
        return os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    
    @property
    def model_registry_uri(self) -> str:
        """
        Retorna a URI do registro de modelos.
        
        Returns:
            str: URI do registro de modelos
        """
        return os.getenv("MODEL_REGISTRY_URI", "sqlite:///mlflow.db")