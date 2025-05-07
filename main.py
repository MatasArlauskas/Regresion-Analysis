import pandas as pd
import zipfile

# Atidaro ZIP failą
with zipfile.ZipFile('/content/drive/MyDrive/Colab Notebooks/realtor-data.zip.csv.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/')

# Nuskaityk CSV failą
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/realtor-data.zip.csv.zip')

# Peržiūrėk pirmas eilutes
df.head()


# Struktūra
df.info()

# Statistinė suvestinė
df.describe()

# Trūkstamų reikšmių analizė
df.isnull().sum()

# Unikalios reikšmės stulpeliuose (pvz. status, state)
df.nunique()

# RODO KIEK TRUKSTAMU REIKSMIU YRA KIEKVIENAME STULPELYJE

# Pašalinam city ir zip_code
df_reduced = df.drop(columns=['brokered_by', 'status', 'street', 'prev_sold_date', 'city', 'zip_code'])

# Pašalinam įrašus, kur trūksta svarbių stulpelių
df_reduced = df_reduced.dropna(subset=['price', 'bed', 'bath', 'acre_lot', 'house_size'])

# Patikrinam likusias trūkstamas reikšmes
df_reduced.isnull().sum()

# MATOME KAD RODO NULIUS VADINASI NEBELIKO TRUKSTAMU REIKSMIU

# One-hot koduojame 'state' (regioną)
df_encoded = pd.get_dummies(df_reduced, columns=['state'], drop_first=True)

# ZODZIUS state STULPELYJE PAVERCIAME SKAICIAIS, taip lengviau skaiciuoti korealiacija

from sklearn.preprocessing import StandardScaler

# Pasirenkame skaitinius požymius
num_features = ['bed', 'bath', 'acre_lot', 'house_size']

# Kuriame skalę
scaler = StandardScaler()

# Priskiriame transformuotas reikšmes
df_encoded[num_features] = scaler.fit_transform(df_encoded[num_features])

# SKAITINIU VERCIU SUVIENODINIMAS (JU MASTELIO

import matplotlib.pyplot as plt

df_encoded[['price', 'bed', 'bath', 'acre_lot', 'house_size']].hist(bins=30, figsize=(10, 6))
plt.tight_layout()
plt.suptitle('Skaitinių kintamųjų pasiskirstymas', y=1.02)
plt.show()

# DUOMENYS - dominuoja mazos vertes ir labai dideles reiksmes OUTLIERIES 

import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(12, 8))
sns.heatmap(df_encoded.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Koreliacijos matrica")
plt.show()

# MATOME KAD MODELIS TURETU LABIAUSIAI PASIKLIAUTI size, bath, bed

"""
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(16, 10))
sns.heatmap(df_final.corr(numeric_only=True), cmap='coolwarm', center=0, cbar=True)
plt.title("Koreliacijos matrica (be reikšmių)")
plt.show()
"""

from scipy import stats

z_scores = stats.zscore(df_encoded[['price', 'bed', 'bath', 'acre_lot', 'house_size']])
abs_z_scores = abs(z_scores)
filtered_entries = (abs_z_scores < 4).all(axis=1)

# Pritaikome filtrą
df_final = df_encoded[filtered_entries]

# PASALINA OUTLIERS


# Duomenu padalijimas i mokymo ir testavimo rinkinius
from sklearn.model_selection import train_test_split

# X – požymiai (visi išskyrus 'price')
X = df_final.drop(columns=['price'])
y = df_final['price']

# 80% mokymui, 20% testavimui
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# SUKURIAMAS IR APMOKAMAS TIESINES REGRESIJOS MODELIS

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Modelio sukūrimas ir apmokymas
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Prognozė
y_pred_lr = lin_reg.predict(X_test)

# Įvertinimas
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Linear Regression → MAE: {mae_lr:.2f}, MSE: {mse_lr:.2f}, R2: {r2_lr:.2f}")

# MAE (Mean Absolute Error) - vidutine absoliuti klaida tarp prognozuojamos ir tikros kainos. Kuo mazesne klaida tuo tiksliau.
# MSE (Mean Squared Error) - kvadratine klaida, labai jautri outlieriems, modelis labai klysta.
# R2 (Determination coeficient) - 1 yra geras rezultatas.

X_train.shape  # (eilutės, stulpeliai)



# Pvz. išsirenkam tik svarbiausius požymius
selected_features = ['house_size', 'bath', 'bed']  # arba kiti pagal koreliaciją

X_train_poly = X_train[selected_features]
X_test_poly = X_test[selected_features]



# POLINOMINES REGRESIJOS MODELIS

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(X_train_poly, y_train)

y_pred_poly = poly_model.predict(X_test_poly)

mae_poly = mean_absolute_error(y_test, y_pred_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print(f"Polynomial Regression → MAE: {mae_poly:.2f}, MSE: {mse_poly:.2f}, R2: {r2_poly:.2f}")


# SPRENDIMU MEDZIO MODELIS

from sklearn.tree import DecisionTreeRegressor

# Sukuriam modelį su numatytais hiperparametrais
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

# Prognozė
y_pred_tree = tree_model.predict(X_test)

# Įvertinimas
mae_tree = mean_absolute_error(y_test, y_pred_tree)
mse_tree = mean_squared_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)

print(f"Decision Tree → MAE: {mae_tree:.2f}, MSE: {mse_tree:.2f}, R2: {r2_tree:.2f}")


# MISKO MODELIS

from sklearn.ensemble import RandomForestRegressor

# Naudojam 100 medžių (gali būti daugiau, bet ilgiau mokys)
forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
forest_model.fit(X_train, y_train)

# Prognozė
y_pred_forest = forest_model.predict(X_test)

# Įvertinimas
mae_forest = mean_absolute_error(y_test, y_pred_forest)
mse_forest = mean_squared_error(y_test, y_pred_forest)
r2_forest = r2_score(y_test, y_pred_forest)

print(f"Random Forest → MAE: {mae_forest:.2f}, MSE: {mse_forest:.2f}, R2: {r2_forest:.2f}")



#paruosiame duomenis RNN modeliui
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.preprocessing import MinMaxScaler

# Požymiai RNN modeliui
features = ['house_size', 'bath', 'bed', 'acre_lot']
X_rnn = df_final[features]
y_rnn = df_final['price']

# Normalizuojam
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_x.fit_transform(X_rnn)
y_scaled = scaler_y.fit_transform(y_rnn.values.reshape(-1, 1))

# Sekų kūrimas
def create_sequences(X, y, window_size):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size])
    return np.array(X_seq), np.array(y_seq)

window_size = 10
X_seq, y_seq = create_sequences(X_scaled, y_scaled, window_size)

# Padalinam į mokymo ir testavimo duomenis
split = int(0.8 * len(X_seq))
X_train_rnn, X_test_rnn = X_seq[:split], X_seq[split:]
y_train_rnn, y_test_rnn = y_seq[:split], y_seq[split:]


#Sukuriame ir apmokome RNN modeli

from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(SimpleRNN(50, activation='tanh', input_shape=(window_size, len(features))))
model.add(Dense(1))

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

history = model.fit(X_train_rnn, y_train_rnn, epochs=20, batch_size=32, validation_split=0.2)


# modelio ivertinimas
# Prognozė
y_pred_rnn = model.predict(X_test_rnn)

# Grąžinam originalų mastelį
y_test_inv = scaler_y.inverse_transform(y_test_rnn)
y_pred_inv = scaler_y.inverse_transform(y_pred_rnn)

# Skaičiuojam metrikas
mae_rnn = mean_absolute_error(y_test_inv, y_pred_inv)
mse_rnn = mean_squared_error(y_test_inv, y_pred_inv)
r2_rnn = r2_score(y_test_inv, y_pred_inv)

print(f"RNN → MAE: {mae_rnn:.2f}, MSE: {mse_rnn:.2f}, R2: {r2_rnn:.2f}")


import pandas as pd

results = pd.DataFrame({
    'Modelis': ['Tiesinė regresija', 'Polinominė regresija', 'Sprendimų medis', 'Atsitiktiniai miškai', 'RNN'],
    'MAE': [mae_lr, mae_poly, mae_tree, mae_forest, mae_rnn],
    'MSE': [mse_lr, mse_poly, mse_tree, mse_forest, mse_rnn],
    'R2': [r2_lr, r2_poly, r2_tree, r2_forest, r2_rnn]
})

results


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_forest, alpha=0.5)
plt.xlabel("Tikros kainos")
plt.ylabel("Prognozuotos kainos")
plt.title("Prognozė vs Faktinė kaina – Random Forest")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()


residuals = y_test - y_pred_forest

plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=50)
plt.title("Klaidų pasiskirstymo histograma – Random Forest")
plt.xlabel("Klaida (faktinė - prognozuota)")
plt.ylabel("Dažnis")
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title("RNN mokymo kreivės")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.show()


