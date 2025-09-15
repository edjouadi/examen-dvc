import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import os


def train_model(params_file, X_train_file, y_train_file, output_file):
    """
    Trains the model using the best hyperparameters found.
    """
    best_params = joblib.load(params_file)
    X_train = pd.read_csv(X_train_file)
    y_train = pd.read_csv(y_train_file)
    
    model = GradientBoostingRegressor(**best_params)
    model.fit(X_train, y_train.values.ravel())
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    joblib.dump(model, output_file)
    print("Model trained and saved.")


if __name__ == '__main__':
    train_model(
        os.path.join('models', 'best_params.pkl'),
        os.path.join('data', 'processed', 'X_train_scaled.csv'),
        os.path.join('data', 'processed', 'y_train.csv'),
        os.path.join('models', 'gbr_model.pkl')
    )