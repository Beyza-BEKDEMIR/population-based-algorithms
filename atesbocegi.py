import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time
from sklearn.metrics import mean_squared_error

# Veri setini okuyalım
file_path = 'world_happiness_report.csv'
data = pd.read_csv(file_path, sep=',')

# Kategorik sütunları Label Encoding ile dönüştürme
label_encoder = LabelEncoder()
categorical_columns = ['Country_or_region']
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])
    
# NaN değerleri median ile doldur
data = data.fillna(data.median())

# Veriyi karıştır
data = data.sample(frac=1, random_state=25).reset_index(drop=True)

# Tüm sütunları scores değişkenine ata
scores = data[['Overall_rank','Country_or_region','Score','GDP','Social_support','Healthy_life','Freedom_to_make_life_choices','Generosity','Perceptions_of_corruption']].values

# En iyi satırı bulma
best_row_index = np.argmax(scores[:, -7:].sum(axis=1))

# En iyi satırın gerçek skorlarını al
true_scores = scores[best_row_index, -7:]

# Fitness fonksiyonunu tanımlama
def fitness_function(position):
    return np.sum(position[-7:])  

# Başlangıç pozisyonlarını belirleme fonksiyonu
def initialize_positions(scores):
    n, dim = scores.shape
    return np.random.uniform(0, 10, size=(n, dim))

# Ateş Böceği Algoritması
def firefly_algorithm(scores, max_iter):
    n, dim = scores.shape
    # Başlangıç pozisyonları
    positions = initialize_positions(scores)
    best_position = positions[0].copy()
    best_fitness = fitness_function(best_position)
    
    # Ateşböceği algoritması için parametreler
    alpha = 0.5  # Alfa parametresi
    beta = 0.2   # Beta parametresi
    gamma = 1    # Gama parametresi
    
    start_time = time.time()  # Zamanı kaydetmek için başlangıç zamanı
    
    for t in range(max_iter):
        for i in range(n):
            for j in range(n):
                if fitness_function(positions[j]) > fitness_function(positions[i]):
                    attractiveness = alpha * np.exp(-gamma * np.linalg.norm(positions[i] - positions[j])**2)  # Çekim gücü hesabı
                    positions[i] += attractiveness * (positions[j] - positions[i]) + beta * (np.random.rand(dim) - 0.5)  # Yeni konum
                    positions[i] = np.clip(positions[i], 0, 10)  # Pozisyonları sınırla

            # En iyi çözümü güncelle
            if fitness_function(positions[i]) > best_fitness:
                best_fitness = fitness_function(positions[i])
                best_position = positions[i].copy()
                        
        # Her iterasyonda en iyi pozisyonu ve fitness değerini yazdır
        print(f"Iter: {t+1}\nEn İyi Fitness: {best_fitness},\nEn İyi Pozisyon: {best_position[-7:]}")
    
    elapsed_time = time.time() - start_time  # Hesaplama süresini hesapla
    
    print("\n---------------------------------------------------\n")
    
    print("En iyi satır indeksi:", best_row_index)
    print("true_scores:",true_scores)
    print(f"En İyi Pozisyon: {best_position[-7:]},\nEn İyi Fitness: {best_fitness}")
    print(f"Hesaplama Süresi: {elapsed_time} saniye")

    predicted_scores = best_position[-7:]  # En iyi satırın tahmin edilen skorları
    
    # MSE ve RMSE hesapla
    mse = mean_squared_error(true_scores, predicted_scores)
    rmse = np.sqrt(mse)
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    
    # En iyi pozisyonu ve en iyi satır indeksini döndür
    return best_position, best_fitness

# Parametreler
max_iter = 20  # Maksimum iterasyon sayısı

# Ateş Böceği Algoritması'nı çalıştırma
best_position, best_fitness = firefly_algorithm(scores, max_iter)