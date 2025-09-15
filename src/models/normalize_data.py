import pandas as pd
from sklearn.preprocessing import StandardScaler
import os


def normalize_data(input_dir, output_dir):
    """
    Normalizes the training and testing data using StandardScaler.
    """
    X_train = pd.read_csv(os.path.join(input_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(input_dir, 'X_test.csv'))
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaled data as DataFrames
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    os.makedirs(output_dir, exist_ok=True)
    X_train_scaled_df.to_csv(os.path.join(output_dir, 'X_train_scaled.csv'), index=False)
    X_test_scaled_df.to_csv(os.path.join(output_dir, 'X_test_scaled.csv'), index=False)


if __name__ == '__main__':
    normalize_data(os.path.join('data', 'processed'), os.path.join('data', 'processed'))