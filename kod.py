# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 18:21:05 2024

@author: Müleyke
"""
import pandas as pd
import numpy as np

file_path = '/Users/cm/Desktop/Verı Madencılıgı/Ara SInav/orjinal_veri.xlsx'
data = pd.read_excel(file_path)

data.info()
data.dtypes

#Analiz için önemsiz olduğunu düşündüğümüz değişkenleri kaldırdık. (Feature selection
data = data.drop('varsakımde ANNE', axis=1)
data = data.drop('varsakımde BABA', axis=1)
data = data.drop('varsakımde KARDES', axis=1)
data = data.drop('varsakımde DİĞER', axis=1)
##Bu iki niteliğin 0'dan başka hiçbir değeri yok. Yani tüm değerleri 0 olduğu için anlamsız nitelikler. Veriden çıkaralım.
data = data.drop('yogumbakımatoplamyatıssuresısaat', axis=1)
data = data.drop('servıseoplamyatıssuresısaat', axis=1)
#%30'dan fazla eksik veri içerdiği için bu nitelikleri veriden çıkıyoruz.
data = data.drop('sıgarayıbırakannekadarGÜNıcmıs', axis=1)
data = data.drop('sıgarabırakangundekacadetıcmıs', axis=1)
data = data.drop('nezamanbırakmısGÜN', axis=1)
data = data.drop('sıgarayadevamedengundekacadetıcıyo', axis=1)
data = data.drop('FEV1', axis=1)
data = data.drop('PEF', axis=1)

#Vki değişkenini ekle
data["boy"] = data["boy"] / 100
data["VKİ"] = data["vucutagırlıgı"] / (data["boy"] ** 2)
print(data.head())
data = data.drop('boy', axis=1)
data = data.drop('vucutagırlıgı', axis=1)

#kategorik ve sürekli değişkenleri belirleyelim.

data["cınsıyet"] = data["cınsıyet"].astype("category")
data["meslegı"] = data["meslegı"].astype("category")
data["egıtımduzeyı"] = data["egıtımduzeyı"].astype("category")
data["sıgarakullanımı"] = data["sıgarakullanımı"].astype("category")
data["tanı"] = data["tanı"].astype("category")
data["hastaneyeyattımı"] = data["hastaneyeyattımı"].astype("category")
data["ailedekoahveyaastımTanılıHastavarmı"] = data["ailedekoahveyaastımTanılıHastavarmı"].astype("category")
#Kategorik değişkenlerin değerleninin tanımlanması

data["cınsıyet"].value_counts()
data["cınsıyet"] = data["cınsıyet"].replace({1: "ERKEK", 2: "KADIN"})
data["egıtımduzeyı"].value_counts()
data["egıtımduzeyı"] = data["egıtımduzeyı"].replace({1: "YÜKSEKOKUL", 2: "LİSE", 3:"İLKOKUL", 4:"OKUR-YAZAR", 5:"OKUR-YAZAR DEĞİL"})
data["meslegı"].value_counts()
data["meslegı"] = data["meslegı"].replace({1: "İŞSİZ", 2: "EMEKLİ", 3:"MEMUR", 4:"İŞÇİ", 5:"ÖZEL SEKTÖR", 6:"SERBEST"})
data["sıgarakullanımı"].value_counts()
data["sıgarakullanımı"] = data["sıgarakullanımı"].replace({1: "HİÇ İÇMEMEMİŞ", 2: "BIRAKMIŞ", 3:"HALEN İÇİYOR"})

# 'tanı' sütunundaki 1'i 0'a, 2'yi 1'e dönüştürme
data["tanı"] = data["tanı"].replace({2: 0})
data["tanı"].value_counts()
data["tanı"] = data["tanı"].replace({0: "KOAH", 1: "ASTIM"})
data["hastaneyeyattımı"].value_counts()
data["hastaneyeyattımı"] = data["hastaneyeyattımı"].replace({1: "HAYIR", 2: "EVET"})
data["ailedekoahveyaastımTanılıHastavarmı"].value_counts()
data["ailedekoahveyaastımTanılıHastavarmı"] = data["ailedekoahveyaastımTanılıHastavarmı"].replace({1: "HAYIR", 2: "EVET"})


data = data.replace("na", np.nan)


#Kontrol 
data.info()
data.dtypes

#Eksik gözlem olup olmadığını kontrol edelim
data.isnull().any()
data.isnull().sum()

#eksik gözlem içeren niteliklerin eksik kısımlarını niteliğin medyan değerlerini kullanarak dolduralım
data['kanbasıncıdıastolık'].fillna(data['kanbasıncıdıastolık'].median(), inplace=True)
data['kanbasıncısıstolık'].fillna(data['kanbasıncısıstolık'].median(), inplace=True)
data['PEF %'].fillna(data['PEF %'].median(), inplace=True)
data['tanısuresıyıl'].fillna(data['tanısuresıyıl'].median(), inplace=True)
data['tanısuresıay'].fillna(data['tanısuresıay'].median(), inplace=True)


kategorik_degiskenler = data.select_dtypes(include=['category'])


numerik_degiskenler = data.select_dtypes(include=['float64', 'int64'])



import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Verilerinizi standartlaştırma
def standardize_data(df):
    # Numerik sütunları seç
    numerik_degiskenler = df.select_dtypes(include=[np.number]).columns
    
    # Standartlaştırma
    scaler = StandardScaler()
    df[numerik_degiskenler] = scaler.fit_transform(df[numerik_degiskenler])
    
    return df

# Aykırı değerleri tespit eden ve temizleyen fonksiyon
def remove_outliers_zscore(df, threshold=3):
    # Numerik sütunları seç
    numerik_degiskenler = df.select_dtypes(include=[np.number]).columns
    
    # Z skorlarını hesapla
    z_scores = np.abs(stats.zscore(df[numerik_degiskenler]))
    
    # Aykırı değer olmayan satırları seç
    no_outliers = ~(z_scores > threshold).any(axis=1)
    cleaned_df = df[no_outliers]
    
    print(f"Orijinal veri seti boyutu: {df.shape}")
    print(f"Aykırı değerler silindikten sonra veri seti boyutu: {cleaned_df.shape}")
    return cleaned_df

# Örnek kullanım:
# DataFrame'inizi okuyun veya oluşturun
# Örneğin, 'data' adında bir DataFrame'iniz var

# Standartlaştırma
standardized_data = standardize_data(data)

# Aykırı değerleri temizleme
cleaned_data = remove_outliers_zscore(standardized_data)

import pandas as pd

# Kategorik değişkenleri seçme
kategorik_degiskenler = cleaned_data.select_dtypes(include=['object', 'category']).columns

# One-Hot Encoding uygulama
encoded_data = pd.get_dummies(cleaned_data, columns=kategorik_degiskenler, drop_first=True)

# Sonuç
print("One-Hot Encoding uygulanmış veri:")
print(encoded_data.head())


X = encoded_data.drop('tanı_KOAH', axis=1)  # 'hedef_değişken_adı' hedef değişkeninizi temsil eder
y = encoded_data['tanı_KOAH']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=42)

model.fit(X_train, y_train)

print(X.isnull().sum())  # Her sütundaki eksik değer sayısını gösterir


##Test kümesi işlemleri 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=42)

X_train = np.array(X_train)
y_train = np.array(y_train)

model.fit(X_train, y_train)

print(X.isnull().sum())  # Her sütundaki eksik değer sayısını gösterir

y_pred = model.predict(X_test)


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

from sklearn.metrics import classification_report

print("Classification Report:")
print(classification_report(y_test, y_pred))


train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")

import statsmodels.api as sm

# Bağımsız (X) ve bağımlı (y) değişkenler
encoded_data = encoded_data.astype(float)
X = encoded_data.drop('tanı_KOAH', axis=1)
y = encoded_data['tanı_KOAH']


# Sabit terim ekleme
X = sm.add_constant(X)

# Lojistik regresyon modeli
logit_model = sm.Logit(y, X)
result = logit_model.fit()

# Katsayılar (β) ve Odds Oranları (Exp(B))
coefficients = result.params  # Katsayılar
odds_ratios = np.exp(result.params)  # Odds oranları

# Sonuçları göster
print("Katsayılar (β):")
print(coefficients)
print("\nOdds Oranları (Exp(β)):")
print(odds_ratios)

# Model özeti
print(result.summary())


import joblib

# Modeli kaydetme
joblib.dump(model, 'lojistik_regresyon_modeli.pkl')

# Daha sonra modeli yüklemek için
model = joblib.load('lojistik_regresyon_modeli.pkl')



import pandas as pd
import numpy as np

file_path = '/Users/cm/Desktop/Verı Madencılıgı/Ara SInav/test_veri.xlsx'
data_test = pd.read_excel(file_path)


data_test = data_test.drop('hastaNo', axis=1)
data_test = data_test.drop('basvurutarıhı', axis=1)

data_test = data_test.drop('varsakımde ANNE', axis=1)
data_test = data_test.drop('varsakımde BABA', axis=1)
data_test = data_test.drop('varsakımde KARDES', axis=1)
data_test = data_test.drop('varsakımde DİĞER', axis=1)

data_test = data_test.drop('yogumbakımatoplamyatıssuresısaat', axis=1)
data_test = data_test.drop('servıseoplamyatıssuresısaat', axis=1)
#%30'dan fazla eksik veri içerdiği için bu nitelikleri veriden çıkıyoruz.
data_test = data_test.drop('sıgarayıbırakannekadarGÜNıcmıs', axis=1)
data_test = data_test.drop('sıgarabırakangundekacadetıcmıs', axis=1)
data_test = data_test.drop('nezamanbırakmısGÜN', axis=1)
data_test = data_test.drop('sıgarayadevamedengundekacadetıcıyo', axis=1)
data_test = data_test.drop('FEV1', axis=1)
data_test = data_test.drop('PEF', axis=1)

data_test["boy"] = data_test["boy"] / 100
data_test["VKİ"] = data_test["vucutagırlıgı"] / (data_test["boy"] ** 2)
print(data_test.head())
data_test = data_test.drop('boy', axis=1)
data_test = data_test.drop('vucutagırlıgı', axis=1)

data_test["cınsıyet"] = data_test["cınsıyet"].astype("category")
data_test["meslegı"] = data_test["meslegı"].astype("category")
data_test["egıtımduzeyı"] = data_test["egıtımduzeyı"].astype("category")
data_test["sıgarakullanımı"] = data_test["sıgarakullanımı"].astype("category")
data_test["hastaneyeyattımı"] = data_test["hastaneyeyattımı"].astype("category")
data_test["ailedekoahveyaastımTanılıHastavarmı"] = data_test["ailedekoahveyaastımTanılıHastavarmı"].astype("category")


data_test["cınsıyet"].value_counts()
data_test["cınsıyet"] = data_test["cınsıyet"].replace({1: "ERKEK", 2: "KADIN"})
data_test["egıtımduzeyı"].value_counts()
data_test["egıtımduzeyı"] = data_test["egıtımduzeyı"].replace({1: "YÜKSEKOKUL", 2: "LİSE", 3:"İLKOKUL", 4:"OKUR-YAZAR", 5:"OKUR-YAZAR DEĞİL"})
data_test["meslegı"].value_counts()
data_test["meslegı"] = data_test["meslegı"].replace({1: "İŞSİZ", 2: "EMEKLİ", 3:"MEMUR", 4:"İŞÇİ", 5:"ÖZEL SEKTÖR", 6:"SERBEST"})
data_test["sıgarakullanımı"].value_counts()
data_test["sıgarakullanımı"] = data_test["sıgarakullanımı"].replace({1: "HİÇ İÇMEMEMİŞ", 2: "BIRAKMIŞ", 3:"HALEN İÇİYOR"})
data_test["hastaneyeyattımı"].value_counts()
data_test["hastaneyeyattımı"] = data_test["hastaneyeyattımı"].replace({1: "HAYIR", 2: "EVET"})
data_test["ailedekoahveyaastımTanılıHastavarmı"].value_counts()
data_test["ailedekoahveyaastımTanılıHastavarmı"] = data_test["ailedekoahveyaastımTanılıHastavarmı"].replace({1: "HAYIR", 2: "EVET"})


data_test = data_test.replace("na", np.nan)
#BURDA ONUR HOCA 80,6 OLAN BİR DEĞERİ METİN OLARAK EKLEMİŞ O YÜZDEN HATA VERİYORDU BİZ ONU EKSELDE NUMERİK VERİYE ÇEVİRİP TEKRAR OKUTTUK VERİYİ

data_test.info()
data_test.dtypes

data_test.isnull().any()
data_test.isnull().sum()

data_test['PEF %'].fillna(data_test['PEF %'].median(), inplace=True)
data_test['FEV1 %'].fillna(data_test['FEV1 %'].median(), inplace=True)
data_test['ailedekoahveyaastımTanılıHastavarmı'] = data_test['ailedekoahveyaastımTanılıHastavarmı'].fillna(data_test['ailedekoahveyaastımTanılıHastavarmı'].mode()[0])
data_test['tanısuresıay'].fillna(data_test['tanısuresıay'].median(), inplace=True)
data_test['YAŞ'].fillna(data['YAŞ'].median(), inplace=True)

kategorik_degiskenler_test = data_test.select_dtypes(include=['category'])

numerik_degiskenler_test = data_test.select_dtypes(include=['float64', 'int64'])


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Verilerinizi standartlaştırma
def standardize_data(df):
    # Numerik sütunları seç
    numerik_degiskenler_test = df.select_dtypes(include=[np.number]).columns
    
    # Standartlaştırma
    scaler = StandardScaler()
    df[numerik_degiskenler_test] = scaler.fit_transform(df[numerik_degiskenler_test])
    
    return df

standardized_data_test = standardize_data(data_test)


import pandas as pd

# Kategorik değişkenleri seçme
kategorik_degiskenler_test = standardized_data_test.select_dtypes(include=['object', 'category']).columns

# One-Hot Encoding uygulama
encoded_data_test = pd.get_dummies(standardized_data_test, columns=kategorik_degiskenler_test, drop_first=True)

# Sonuç
print("One-Hot Encoding uygulanmış veri:")
print(encoded_data_test.head())

y_pred = model.predict(encoded_data_test)

encoded_data_test['tanı'] = y_pred

print(encoded_data_test)

