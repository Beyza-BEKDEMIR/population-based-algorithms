import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import time

# Veri setini yükle
data = pd.read_csv("world_happiness_report.csv")

# Kategorik değişkeni sayısal hale dönüştür
data['Country_or_region'] = LabelEncoder().fit_transform(data['Country_or_region'])

# Fill NaN values with median
data = data.fillna(data.median())

# Shuffle the data
data = data.sample(frac=1, random_state=25).reset_index(drop=True)

# Bağımsız değişkenler (X) ve bağımlı değişken (y) olarak ayır
X = data[['Overall_rank','Country_or_region','Score','GDP','Social_support','Healthy_life','Freedom_to_make_life_choices','Generosity','Perceptions_of_corruption']].values
y = data['Score'].values


def genetic_algorithm(X, y, population_size=156, num_generations=50, mutation_rate=0.3):
    def fitness_function(chromosome):
        return np.sum(chromosome[-7:])

    def crossover(parent1, parent2):
        # Üç noktalı çaprazlama
        crossover_points = sorted(np.random.choice(len(parent1) - 1, size=3, replace=False))
        child1 = np.concatenate((parent1[:crossover_points[0]],
                                  parent2[crossover_points[0]:crossover_points[1]],
                                  parent1[crossover_points[1]:crossover_points[2]],
                                  parent2[crossover_points[2]:]))
        child2 = np.concatenate((parent2[:crossover_points[0]],
                                  parent1[crossover_points[0]:crossover_points[1]],
                                  parent2[crossover_points[1]:crossover_points[2]],
                                  parent1[crossover_points[2]:]))
        return child1, child2

    def mutation(child):
        for i in range(len(child)):
            if np.random.rand() < mutation_rate:
                child[i] = 1 - child[i]  # Geni mutasyona uğrat
        return child

    num_features = X.shape[1]
    population = np.random.randint(2, size=(population_size, num_features))

    best_individual = None
    best_fitness = float('-inf')
    best_index = None

    start_time = time.time()

    best_individuals = []

    for generation in range(num_generations):
        fitness_values = np.apply_along_axis(fitness_function, 1, population)

        best_index = np.argmax(fitness_values)  # En iyi bireyin indisini al
        best_individual = population[best_index].copy()
        best_individuals.append(best_individual)

        # En iyi çözümü güncelle
        if fitness_values[best_index] > best_fitness:
            best_fitness = fitness_values[best_index]
            best_individual = population[best_index].copy()

        new_population = []

        num_elites = population_size // 10
        elite_indices = np.argsort(fitness_values)[-num_elites:]
        elites = population[elite_indices]
        new_population.extend(elites)

        while len(new_population) < population_size:
            parent1, parent2 = population[np.random.choice(np.arange(population_size), size=2, replace=True)]
            child1, child2 = crossover(parent1, parent2)
            child1 = mutation(child1) if np.random.rand() < mutation_rate else child1
            child2 = mutation(child2) if np.random.rand() < mutation_rate else child2
            new_population.extend([child1, child2])

        population = np.array(new_population[:population_size])

        print(f"Generation {generation}: Best Fitness = {best_fitness}")
        print(f"Best Individual: {best_individual}")
        print(f"Best Individual's Row Index: {best_index}")

    elapsed_time = time.time() - start_time

    best_individual_row_index = np.argmax(fitness_values)  # Son iterasyonda en iyi bireyin satır indeksi
    best_individual = population[best_individual_row_index]  # Son iterasyonda en iyi birey
    
    # En iyi bireyin tüm sütun değerlerini tahmin edilen skorlar olarak al
    predicted_scores = X[best_individual_row_index, -7:]
    
    # Gerçek skorları al
    true_scores = X[np.argmax(X[:, -7:].sum(axis=1)), -7:]
    
    # MSE ve RMSE hesapla
    mse = mean_squared_error(true_scores, predicted_scores)
    rmse = np.sqrt(mse)
    
    # Sonuçları yazdır
    print("\n")
    print("True Scores:", true_scores)
    print("Predicted Scores:", predicted_scores)
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Hesaplama Süresi: {elapsed_time} saniye")
    
    # Tahmin edilen ve gerçek skorları yan yana yazdır
    print("\nTahmin Edilen ve Gerçek Skorlar:")
    print("Tahmin Edilen    Gerçek")
    for predicted, true in zip(predicted_scores, true_scores):
        print(f"{predicted:.3f}          {true:.3f}")
        
    return best_individual, best_fitness, elapsed_time

# Genetik algoritmayı çalıştır
best_individual, best_fitness, elapsed_time = genetic_algorithm(X, y)
