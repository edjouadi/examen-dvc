import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import json
import joblib
import os


def evaluate_model(model_file, X_test_file, y_test_file, metrics_file, predictions_file):
    """
    Evaluates the trained model, saves metrics and predictions.
    """
    model = joblib.load(model_file)
    X_test = pd.read_csv(X_test_file)
    y_test = pd.read_csv(y_test_file)
    
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    scores = {
        "mse": mse,
        "r2": r2
    }
    
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    with open(metrics_file, "w") as f:
        json.dump(scores, f, indent=4)
        
    # Create a DataFrame for predictions
    predictions_df = pd.DataFrame({
        'X_test': X_test.to_dict('records'),
        'y_test': y_test.values.flatten(),
        'predictions': predictions
    })
    
    os.makedirs(os.path.dirname(predictions_file), exist_ok=True)
    predictions_df.to_csv(predictions_file, index=False)
    
    print(f"Model evaluation completed. Metrics saved to {metrics_file}, predictions saved to {predictions_file}.")


if __name__ == '__main__':
    evaluate_model(
        os.path.join('models', 'gbr_model.pkl'),
        os.path.join('data', 'processed', 'X_test_scaled.csv'),
        os.path.join('data', 'processed', 'y_test.csv'),
        os.path.join('metrics', 'scores.json'),
        os.path.join('data', 'prediction.csv')
    )