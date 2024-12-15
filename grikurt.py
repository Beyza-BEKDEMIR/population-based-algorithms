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


# Fitness fonksiyonunu tanımlama
def fitness_function(position):
    total = np.sum(position[-7:])
    return total

# Gri Kurt Optimizasyon Algoritması
def gwo_optimization_algorithm(scores, max_iter):
    n, dim = scores.shape  # Veri setindeki örnek sayısı ve boyut sayısı
    # Başlangıç pozisyonları
    positions = np.random.uniform(0, 10, size=scores.shape)  # Pozisyonları 0-10 arasında başlatma (daha dar aralık)
    alpha_pos = np.zeros(dim)  # Alfa kurdunun pozisyonu
    beta_pos = np.zeros(dim)   # Beta kurdunun pozisyonu
    delta_pos = np.zeros(dim)  # Delta kurdunun pozisyonu
    alpha_score = -np.inf  # Alfa kurdunun fitness değeri
    beta_score = -np.inf   # Beta kurdunun fitness değeri
    delta_score = -np.inf  # Delta kurdunun fitness değeri
    
    start_time = time.time()  # Zamanı kaydetmek için başlangıç zamanı

    # İterasyonlar
    for t in range(max_iter):
        for i in range(n):
            fitness = fitness_function(positions[i, :])  # Mevcut pozisyonun fitness değerini hesapla

            # Alfa, Beta ve Delta kurtlarının pozisyonlarını güncelle
            if fitness > alpha_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()
                beta_score = alpha_score
                beta_pos = alpha_pos.copy()
                alpha_score = fitness
                alpha_pos = positions[i, :].copy()
            elif fitness > beta_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()
                beta_score = fitness
                beta_pos = positions[i, :].copy()
            elif fitness > delta_score:
                delta_score = fitness
                delta_pos = positions[i, :].copy()
        
        a = 2 - t * (2 / max_iter)  # a parametresini iterasyon boyunca azaltma
        
        # Her pozisyonu güncelle
        for i in range(n):
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha

                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                X2 = beta_pos[j] - A2 * D_beta

                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                X3 = delta_pos[j] - A3 * D_delta

                positions[i, j] = (X1 + X2 + X3) / 3  # Yeni pozisyonu hesapla

            # Pozisyonları sınırlamak
            positions[i, :] = np.clip(positions[i, :], 0, 7)
        
        # Her iterasyonda en iyi pozisyonları ve fitness değerlerini yazdır
        print(f"Iterasyon: {t+1}")
        print(f"Alfa Pozisyon: {alpha_pos},\nAlfa Fitness: {alpha_score}\n")
        print(f"Beta Pozisyon: {beta_pos},\nBeta Fitness: {beta_score}\n")
        print(f"Delta Pozisyon: {delta_pos},\nDelta Fitness: {delta_score}\n")
    
    elapsed_time = time.time() - start_time  # Hesaplama süresini hesapla
    
    best_row_index = np.argmax(np.sum(scores[:, -7:], axis=1))  # En yüksek toplam not değerine sahip satırın indeksi
    best_row = scores[best_row_index]  # En yüksek toplam not değerine sahip satır

    print("\n--------------------------\n")

    print(f"En İyi Sonucun Olduğu Satır Indeksi:{best_row_index},\nEn İyi Pozisyon:{alpha_pos[-7:]},\nEn İyi Fitness:{alpha_score}")
    print(f"Hesaplama Süresi: {elapsed_time} saniye")
    
    # Doğruluğu hesapla
    predicted_scores = alpha_pos[-7:]  # Sadece Score, GDP, Social_support, Healthy_life, Freedom_to_make_life_choices, Generosity, Perceptions_of_corruption notlarını al

    true_scores = best_row[-7:]  # Gerçek skorları al
    
    # Karşılaştırma işlemi öncesi skorları yazdırarak kontrol edelim
    print(f"Tahmin Edilen Skorlar: {predicted_scores}")
    print(f"Gerçek Skorlar: {true_scores}")
    
    
    # MSE ve RMSE hesapla
    mse = mean_squared_error(true_scores, predicted_scores)
    rmse = np.sqrt(mse)
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    
    return best_row_index, alpha_pos, alpha_score

# Parametreler
max_iter = 20  # Maksimum iterasyon sayısı

# Gri Kurt Optimizasyon Algoritması'nı çalıştırma
best_row, best_position, best_fitness = gwo_optimization_algorithm(scores, max_iter)
