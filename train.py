from tensorflow import keras
import pandas as pd
import joblib
import sklearn.preprocessing as pp
df = pd.read_csv('tsunami.csv')
X_train = df.drop('tsunami', axis=1)
y = df['tsunami']
scaler = pp.MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, 'scaler.pkl')
model = keras.Sequential([
    keras.layers.Dense(64,activation='swish', input_shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(32,activation='swish'),
    keras.layers.Dense(16,activation='swish'),
    keras.layers.Dense(8,activation='swish'),
    keras.layers.Dense(4,activation='swish'),
    keras.layers.Dense(2,activation='swish'),
    keras.layers.Dense(1,activation='sigmoid')
])
model.compile(optimizer='adamax', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_scaled, y, epochs=150, batch_size=32, validation_split=0.2)
model.save('tsunami_model.h5')