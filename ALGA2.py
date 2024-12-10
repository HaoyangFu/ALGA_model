import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os
import random
from scipy.stats import qmc

# File path settings
data_directory = 'D:\\OneDrive\\Machine learning\\Machine learning with small dataset'
original_data_file = os.path.join(data_directory, 'data pool.xlsx')

# Define feature ranges and increments
feature_ranges = [
    (100, 800, 20),  # Calcination temperature (°C)
    (0, 6, 0.5),     # Calcination time (h)
    (0.1, 1, 0.1),   # Calcination heating rate (°C/min)
    (5, 20, 1),      # The added amount of N (wt. %)
    (0, 6, 0.5),     # The added amount of Cu (wt. %)
    (100, 800, 20),  # Pyrolysis temperature (°C)
    (0, 6, 0.5),     # Pyrolysis time (h)
    (0.1, 1, 0.1)    # Pyrolysis heating rate (°C/min)
]

def clean_and_select_data(data, feature_columns, k_column):
    return data[feature_columns + [k_column]]

def conduct_experiment(x):
    while True:
        try:
            print("Please manually input the k value obtained under this condition:")
            k = float(input())
            return k
        except ValueError:
            print("Invalid input. Please enter a number.")

def select_best_params(X, y, rf_params):
    rf_model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf_model, param_grid=rf_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

def crossover(parent1, parent2, feature_ranges):
    child1, child2 = parent1.copy(), parent2.copy()
    for i, (min_val, max_val, increment) in enumerate(feature_ranges):
        if np.random.random() < 0.5:
            child1[i], child2[i] = child2[i], child1[i]
    return child1, child2

def mutate(individual, mutation_rate, feature_ranges):
    for i, (min_val, max_val, increment) in enumerate(feature_ranges):
        if np.random.random() < mutation_rate:
            individual[i] = np.random.choice(np.arange(min_val, max_val + increment, increment))
    return individual

def latin_hypercube_sampling(feature_ranges, num_samples=500):
    mins = np.array([start for start, _, _ in feature_ranges])
    maxs = np.array([end for _, end, _ in feature_ranges])
    sampler = qmc.LatinHypercube(d=len(feature_ranges))
    sample = sampler.random(n=num_samples)
    scaled_sample = qmc.scale(sample, mins, maxs)
    df = pd.DataFrame(scaled_sample, columns=[f'Feature_{i + 1}' for i in range(len(feature_ranges))])
    for i, (_, _, increment) in enumerate(feature_ranges):
        df[f'Feature_{i + 1}'] = np.round(df[f'Feature_{i + 1}'] / increment) * increment
    return df

def optimize(D, P, imax, pc, pm, rf_params, feature_columns, k_column):
    N = len(D)
    i = 0
    scaler = StandardScaler()
    A = latin_hypercube_sampling(feature_ranges, num_samples=500)

    while i < imax:
        print(f"\nIteration {i + 1}:")
        print(f"Current dataset size: {len(D)}")

        X_train, X_test, y_train, y_test = train_test_split(D[feature_columns], D[k_column], test_size=0.2, random_state=42)
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        rf_model = select_best_params(X_train_scaled, y_train, rf_params)

        X_test_scaled = scaler.transform(X_test)
        y_pred = rf_model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Test set MSE: {mse:.4f}")

        A_scaled = scaler.transform(A)
        predictions = rf_model.predict(A_scaled)
        best_index = np.argmax(predictions)
        x_model_best = A.iloc[best_index]

        print("RF recommended best feature values:")
        print(x_model_best.tolist())

        if any(np.allclose(x_model_best, D[feature_columns].iloc[j]) for j in range(len(D))):
            print("RF recommended best feature values are already in the dataset, program terminates.")
            break
        else:
            print("RF recommended best feature values are not in the dataset.")
            print("Do you want to conduct an experiment and add a new k value? (y/n)")
            if input().lower() == 'y':
                print("Please conduct the experiment and input the k value for the recommended conditions:")
                k_model = conduct_experiment(x_model_best)
                new_row = pd.DataFrame({**dict(zip(feature_columns, x_model_best)), k_column: k_model}, index=[0])
                D = pd.concat([D, new_row], ignore_index=True)
                P = pd.concat([P, pd.DataFrame([x_model_best], columns=feature_columns)], ignore_index=True)
                N += 1
            else:
                print("User chose not to add a new k value, program terminates.")
                break

        r = N - (N % 4)  # Largest even number <= N
        R = P.sample(n=r)
        parents = R.sample(frac=0.5)

        new_population = []
        for j in range(0, len(parents), 2):
            if np.random.random() < pc:
                child1, child2 = crossover(parents.iloc[j].tolist(), parents.iloc[j + 1].tolist(), feature_ranges)
                child1 = mutate(child1, pm, feature_ranges)
                child2 = mutate(child2, pm, feature_ranges)
                new_population.extend([child1, child2])

        print("\nGA generated offspring feature values:")
        for idx, child in enumerate(new_population[:2]):
            print(f"Offspring {idx + 1}:", child)
            print(f"Please conduct the experiment and input the k value for offspring {idx + 1}:")
            k_child = conduct_experiment(child)
            new_row = pd.DataFrame({**dict(zip(feature_columns, child)), k_column: k_child}, index=[0])
            D = pd.concat([D, new_row], ignore_index=True)
            P = pd.concat([P, pd.DataFrame([child], columns=feature_columns)], ignore_index=True)
            N += 1

        i += 1

        if i < imax:
            print("\nDo you want to continue to the next iteration? (y/n)")
            if input().lower() != 'y':
                break

    print("\nOptimization process ended.")
    print(f"Total iterations: {i}")
    print(f"Final dataset size: {len(D)}")

    best_index = D[k_column].idxmax()
    return D.loc[best_index, feature_columns], D.loc[best_index, k_column]

# Main execution
original_data = pd.read_excel(original_data_file, header=None)

feature_columns = [f'Feature_{i}' for i in range(1, 9)]
k_column = 'Output'
original_data.columns = feature_columns + [k_column]

seed = random.randint(1, 10000)
random.seed(seed)
np.random.seed(seed)

initial_data = original_data.sample(n=30, random_state=seed)
initial_data = clean_and_select_data(initial_data, feature_columns, k_column)

rf_params = {
    'max_depth': [10, 20, 30, 40, 50],
    'max_leaf_nodes': [50, 100, 150, 200, 250]
}

D = initial_data.copy()
P = D[feature_columns].copy()

imax = 50  # Maximum number of iterations
pc = 0.85  # Crossover probability
pm = 0.03  # Mutation probability

print("Optimization program starts running. Please prepare to input the k values obtained from experiments.")

best_condition, best_k_value = optimize(D, P, imax, pc, pm, rf_params, feature_columns, k_column)

print(f"\nBest conditions: {best_condition.tolist()}")
print(f"Best k value: {best_k_value}")