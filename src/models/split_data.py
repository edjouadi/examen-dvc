import pandas as pd
from sklearn.model_selection import train_test_split
import os


def split_data(input_file, output_dir):
    """
    Splits the raw data into training and testing sets.
    """
    df = pd.read_csv(input_file)
    
    # Separate features and target
    X = df.drop('silica_concentrate', axis=1)
    y = df['silica_concentrate']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save datasets
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    

if __name__ == '__main__':
    split_data('raw.csv', os.path.join('data', 'processed'))