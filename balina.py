import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = 'world_happiness_report.csv'
data = pd.read_csv(file_path, sep=',')

# Label encoding for categorical columns
label_encoder = LabelEncoder()
categorical_columns = ['Country_or_region']
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])
    
# Fill NaN values with median
data = data.fillna(data.median())

# Shuffle the data
data = data.sample(frac=1, random_state=25).reset_index(drop=True)

# Extracting the required columns
scores = data[['Overall_rank', 'Country_or_region', 'Score', 'GDP', 'Social_support', 'Healthy_life', 'Freedom_to_make_life_choices', 'Generosity', 'Perceptions_of_corruption']].values

# En iyi satırı bulma
best_row_index = np.argmax(scores[:, -7:].sum(axis=1))

# En iyi satırın gerçek skorlarını al
true_scores = scores[best_row_index, -7:]

# Objective function (fitness function)
def objective_function(params):
    score_sum = np.sum(params)  # Sum of 'Score', 'GDP', 'Social_support', 'Healthy_life', 'Freedom_to_make_life_choices', 'Generosity', 'Perceptions_of_corruption'
    return -score_sum  # Negative because we want to maximize the score

# Whale Optimization Algorithm (WOA) function
def whale_optimization_algorithm(obj_function, dim, num_agents, max_iter, lb, ub):
    positions = np.random.uniform(0, 1, (num_agents, dim)) * (ub - lb) + lb
    leader_pos = np.zeros(dim)
    leader_score = float("inf")
    
    for t in range(max_iter):
        for i in range(num_agents):
            fitness = obj_function(positions[i, :])
            if fitness < leader_score:
                leader_score = fitness
                leader_pos = positions[i, :].copy()
        
        a = 2 - t * (2 / max_iter)
        for i in range(num_agents):
            r1 = random.random()
            r2 = random.random()
            A = 2 * a * r1 - a
            C = 2 * r2
            p = random.random()
            b = 1
            l = (a - 1) * random.random() + 1
            
            for j in range(dim):
                if p < 0.5:
                    if abs(A) >= 1:
                        rand_leader_index = int(np.floor(num_agents * random.random()))
                        x_rand = positions[rand_leader_index, :]
                        D_x_rand = abs(C * x_rand[j] - positions[i, j])
                        positions[i, j] = x_rand[j] - A * D_x_rand
                    else:
                        D_Leader = abs(C * leader_pos[j] - positions[i, j])
                        positions[i, j] = leader_pos[j] - A * D_Leader
                else:
                    distance2Leader = abs(leader_pos[j] - positions[i, j])
                    positions[i, j] = distance2Leader * np.exp(b * l) * np.cos(l * 2 * np.pi) + leader_pos[j]
                    
        positions = np.clip(positions, lb, ub)
        
        # Print iteration details
        print(f"Iteration {t+1}: Best score = {-leader_score}")

    return leader_pos, leader_score

# WOA parameters
dim = scores.shape[1] - 2  # We exclude 'Overall_rank' and 'Country_or_region'
num_agents = 30
max_iter = 30
lb = np.min(scores[:, 2:], axis=0)
ub = np.max(scores[:, 2:], axis=0)

# Running WOA
best_position, best_score = whale_optimization_algorithm(objective_function, dim, num_agents, max_iter, lb, ub)

print("En iyi satır indeksi:", best_row_index)
print("true_scores:", true_scores)

print("Best position: ", best_position)
print("Best score: ", -best_score)

predicted_scores = best_position  # En iyi satırın tahmin edilen skorları

# MSE ve RMSE hesapla
mse = mean_squared_error(true_scores, predicted_scores)
rmse = np.sqrt(mse)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
