import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import os


def find_best_params(X_train_file, y_train_file, output_file):
    """
    Performs a Grid Search to find the best hyperparameters for the model.
    """
    X_train = pd.read_csv(X_train_file)
    y_train = pd.read_csv(y_train_file)
    
    model = GradientBoostingRegressor()
    
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5]
    }
    
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train.values.ravel())
    
    best_params = grid_search.best_params_
    
    # Save the best parameters to a pkl file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    joblib.dump(best_params, output_file)
    print(f"Best parameters found: {best_params}")


if __name__ == '__main__':
    find_best_params(
        os.path.join('data', 'processed', 'X_train_scaled.csv'), 
        os.path.join('data', 'processed', 'y_train.csv'), 
        os.path.join('models', 'best_params.pkl')
    )