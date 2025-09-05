import pytest
import pandas as pd
from src.data.transform import DataTransformer

class TestDataTransformer:
    def setup_method(self):
        self.transformer = DataTransformer()
    
    def test_clean_data_removes_duplicates(self):
        # Arrange
        df = pd.DataFrame({
            'age': [30, 30, 40],
            'nodes': [1, 1, 2]
        })
        
        # Act
        result = self.transformer.clean_data(df)
        
        # Assert
        assert len(result) == 2
    
    def test_feature_engineering_creates_age_groups(self):
        # Arrange
        df = pd.DataFrame({
            'age': [25, 45, 65],
            'nodes': [1, 2, 3]
        })
        
        # Act
        result = self.transformer.feature_engineering(df)
        
        # Assert
        assert 'age_group' in result.columns
        assert result['age_group'].iloc[0] == 'young'