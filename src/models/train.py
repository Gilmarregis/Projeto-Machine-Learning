import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score

class ModelTrainer:
    def __init__(self):
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("haberman-survival")
    
    def train_with_tracking(self, X_train, X_test, y_train, y_test, model, model_name):
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            mlflow.log_params(model.get_params())
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            
            # Log metrics
            mlflow.log_metrics({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall
            })
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            return model, {'accuracy': accuracy, 'precision': precision, 'recall': recall}