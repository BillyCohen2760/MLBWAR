import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Function to process hitting data
def process_hitting_data(file_path):
    df = pd.read_csv(file_path)
    
    # Convert relevant columns to numeric if necessary
    df = df.apply(pd.to_numeric, errors='ignore')
    
    # Check for missing values
    print("Missing values in each column (Hitting):\n", df.isnull().sum())
    
    # Handle missing values (drop rows with NaNs)
    df = df.dropna()
    
    # Calculate PA/G and apply qualification criteria
    df['PA/G'] = (df['AB'] + df['BB']) / df['GP']
    min_pa_per_game = 3
    df = df[df['PA/G'] >= min_pa_per_game]
    
    # Define features and target
    features = ['R', 'OPS', 'HR', 'K']
    X = df[features]
    y = df['WAR']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Predict WAR on the test set
    y_pred = model.predict(X_test_scaled)
    
    # Print model coefficients
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', ascending=False)
    print("\nFeature Importances (Hitting):\n", feature_importance)
    
    # Print intercept
    print(f'\nIntercept (Hitting): {model.intercept_}')
    
    # Print performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'\nMean Squared Error (Hitting): {mse}')
    print(f'R-squared (Hitting): {r2}')

# Function to process pitching data
def process_pitching_data(file_path):
    df = pd.read_csv(file_path)
    
    # Convert relevant columns to numeric if necessary
    df = df.apply(pd.to_numeric, errors='ignore')
    
    # Check for missing values
    print("Missing values in each column (Pitching):\n", df.isnull().sum())
    
    # Handle missing values (drop rows with NaNs)
    df = df.dropna()
    
    # Apply qualification criteria for SP and RP
    df['Qualifies'] = False
    df.loc[(df['POS'] == 'SP') & (df['GP'] > 0), 'Qualifies'] = df['IP'] > 146
    df.loc[(df['POS'] == 'RP') & (df['GP'] > 0), 'Qualifies'] = df['IP'] >= 49
    
    # Filter data based on qualification
    df = df[df['Qualifies']]
    
    # Drop the 'Qualifies' column as it's no longer needed
    df = df.drop(columns=['Qualifies'])
    
    # Define features and target
    features = ['W', 'ERA', 'WHIP', 'K/9', 'HR', 'IP']
    X = df[features]
    y = df['WAR']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Predict WAR on the test set
    y_pred = model.predict(X_test_scaled)
    
    # Print model coefficients
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', ascending=False)
    print("\nFeature Importances (Pitching):\n", feature_importance)
    
    # Print intercept
    print(f'\nIntercept (Pitching): {model.intercept_}')
    
    # Print performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'\nMean Squared Error (Pitching): {mse}')
    print(f'R-squared (Pitching): {r2}')

# Process both hitting and pitching data
process_hitting_data('MLB Hitting Stats.csv')
process_pitching_data('MLB Pitching Stats.csv')


