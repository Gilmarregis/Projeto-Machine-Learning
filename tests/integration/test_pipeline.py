import pytest
from src.data.extract import DataExtractor
from src.data.transform import DataTransformer
from src.models.train import ModelTrainer

class TestMLPipeline:
    def test_full_pipeline(self, sample_data_path):
        # Extract
        extractor = DataExtractor()
        raw_data = extractor.extract_from_csv(sample_data_path)
        
        # Transform
        transformer = DataTransformer()
        processed_data = transformer.clean_data(raw_data)
        
        # Train
        trainer = ModelTrainer()
        model, metrics = trainer.train_best_model(processed_data)
        
        # Assert
        assert model is not None
        assert metrics['accuracy'] > 0.6