import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
import os

def splitting(features):
    data = pd.read_csv('/Users/sahilnakrani/Documents/weather forecast/src/Machine-Learning/Final-Data/FE_Data.csv')

    if features == 'all':
        X = data[['datetime', 'P', 'RH', 'U']]
        y = data['T']
    elif features == 'date':
        X = data['datetime']
        y = data['T']

    print(X.head())

    cutoff_date = 2022010100

    # Split the data into training and testing sets
    X_train = X[data.datetime < cutoff_date]
    X_test = X[data.datetime >= cutoff_date]

    # Extract temperature values for training and testing
    y_train = y[data.datetime < cutoff_date]
    y_test = y[data.datetime >= cutoff_date]

    # Initialize the StandardScaler
    scaler = StandardScaler()

    if features == 'all':
        # Fit the scaler on the training data and transform both training and testing data
        X_train[['P', 'RH', 'U']] = scaler.fit_transform(X_train[['P', 'RH', 'U']])
        X_test[['P', 'RH', 'U']] = scaler.transform(X_test[['P', 'RH', 'U']])

    # Convert pandas Series to 2D numpy arrays
    X_train = X_train.values.reshape(-1, 1) if features == 'date' else X_train.values
    X_test = X_test.values.reshape(-1, 1) if features == 'date' else X_test.values
    y_train = y_train.to_numpy().reshape(-1, 1)
    y_test = y_test.to_numpy().reshape(-1, 1)

    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test, model_name, features):
    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate the model using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared
    mse = mean_squared_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Print evaluation results
    print(f"Model: {model_name}")
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("R2 Score:", r2)

    # Plot actual vs predicted points
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, predictions)
    plt.xlabel('Actual Temperature')
    plt.ylabel('Predicted Temperature')
    plt.title(f'Actual vs Predicted for {model_name}')

    # Plot date vs temperature
    plt.subplot(1, 2, 2)
    plt.plot(X_test[:, 0], y_test, label='Actual')
    plt.plot(X_test[:, 0], predictions, label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Temperature')
    plt.title(f'Date vs Temperature for {model_name}')
    plt.legend()

    # Save the evaluation results to a text file
    if features == 'all':
        folder_name = f'/Users/sahilnakrani/Documents/weather forecast/src/Machine-Learning/Regression-Models/trained_models/{model_name}/with_all_features'
    elif features == 'date':
        folder_name = f'/Users/sahilnakrani/Documents/weather forecast/src/Machine-Learning/Regression-Models/trained_models/{model_name}/with_only_date'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    with open(os.path.join(folder_name, f'{model_name}_evaluation_results.txt'), 'w') as f:
        f.write(f'Mean Squared Error: {mse}\n')
        f.write(f'Root Mean Squared Error: {rmse}\n')
        f.write(f'R2 Score: {r2}\n')

    # Save the plots
    figure_name = f'{model_name}_evaluation_plots.png'
    if features == 'date':
        figure_name = f'With_only_date_{figure_name}'
    elif features == 'all':
        figure_name = f'With_all_features_{figure_name}'
    plt.savefig(os.path.join(folder_name, figure_name))
    plt.close()