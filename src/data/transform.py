import pandas as pd
import numpy as np
import logging  # ← ADICIONAR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple

class DataTransformer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.logger = logging.getLogger(__name__)
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpeza e tratamento de dados"""
        # Remover duplicatas
        df = df.drop_duplicates()
        
        # Tratar valores faltantes
        df = df.fillna(df.median(numeric_only=True))
        
        # Detectar e tratar outliers
        df = self._remove_outliers(df)
        
        return df
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Criação de novas features"""
        # Criar features categóricas
        df['age_group'] = pd.cut(df['age'], bins=[0, 40, 60, 100], labels=['young', 'middle', 'senior'])
        df['nodes_risk'] = pd.cut(df['nodes'], bins=[-1, 0, 5, 100], labels=['low', 'medium', 'high'])
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers usando IQR"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        return df