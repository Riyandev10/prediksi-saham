# ===============================
# 🔒 WAJIB: SET DETERMINISTIC SEBELUM IMPORT TF
# ===============================
import os
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

from datetime import datetime
import sqlite3
from flask import Flask, request, render_template, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
import tensorflow as tf
import numpy as np
import pandas as pd
import io
import base64
import random
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import GlorotUniform

plt.switch_backend('Agg')

# ===============================
# 🔒 GLOBAL SEED
# ===============================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# ===============================
# FLASK CONFIG
# ===============================
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///prediksi.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'super_secret_key_for_session'
db = SQLAlchemy(app)

MODEL_DIR = os.path.join(os.getcwd(), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ===============================
# DATABASE MODEL
# ===============================
class PrediksiManual(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    previous = db.Column(db.Float, nullable=False)
    high = db.Column(db.Float, nullable=False)
    low = db.Column(db.Float, nullable=False)
    close = db.Column(db.Float, nullable=False)
    prediksi = db.Column(db.Float, nullable=False)
    tanggal = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

# ===============================
# HELPER LOAD MODEL
# ===============================
def load_resources():
    scaler_X = joblib.load(os.path.join(MODEL_DIR, 'scaler_X.pkl'))
    scaler_y = joblib.load(os.path.join(MODEL_DIR, 'scaler_y.pkl'))
    lr_model = joblib.load(os.path.join(MODEL_DIR, 'lr_model.pkl'))
    bp_model = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, 'bp_model.h5'),
        compile=False
    )
    return bp_model, lr_model, scaler_X, scaler_y

# ===============================
# TRAINING FUNCTION
# ===============================
def train_and_compare_models(df):

    X = df[['previous', 'high', 'low']].values
    y = df['close'].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=SEED,
        shuffle=False
    )

    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    joblib.dump(scaler_X, os.path.join(MODEL_DIR, 'scaler_X.pkl'))
    joblib.dump(scaler_y, os.path.join(MODEL_DIR, 'scaler_y.pkl'))

    # ===============================
    # REGRESI LINEAR
    # ===============================
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train_scaled)
    joblib.dump(lr_model, os.path.join(MODEL_DIR, 'lr_model.pkl'))

    lr_pred_test = scaler_y.inverse_transform(
        lr_model.predict(X_test_scaled)
    )
    y_test_original = scaler_y.inverse_transform(y_test_scaled)

    lr_rmse = np.sqrt(mean_squared_error(y_test_original, lr_pred_test))
    lr_mae = mean_absolute_error(y_test_original, lr_pred_test)

    X_all_scaled = scaler_X.transform(X)
    lr_all_pred = scaler_y.inverse_transform(
        lr_model.predict(X_all_scaled)
    )

    # ===============================
    # BACKPROPAGATION NN (DETERMINISTIC)
    # ===============================
    initializer = GlorotUniform(seed=SEED)

    bp_model = Sequential([
        Dense(32, activation='relu',
              kernel_initializer=initializer,
              input_shape=(X_train_scaled.shape[1],)),
        Dense(16, activation='relu', kernel_initializer=initializer),
        Dense(1, kernel_initializer=initializer)
    ])

    bp_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='mse'
    )

    bp_model.fit(
        X_train_scaled,
        y_train_scaled,
        epochs=400,
        batch_size=5,
        verbose=0,
        shuffle=False   
    )

    bp_model.save(os.path.join(MODEL_DIR, 'bp_model.h5'))

    bp_pred_test = scaler_y.inverse_transform(
        bp_model.predict(X_test_scaled, verbose=0)
    )

    bp_rmse = np.sqrt(mean_squared_error(y_test_original, bp_pred_test))
    bp_mae = mean_absolute_error(y_test_original, bp_pred_test)

    bp_all_pred = scaler_y.inverse_transform(
        bp_model.predict(X_all_scaled, verbose=0)
    )

    results = {
        'history': df.to_dict('records'),
        'actual_data': df['close'].tolist(),
        'lr': {
            'predictions': lr_all_pred.flatten().tolist(),
            'rmse': round(lr_rmse, 2),
            'mae': round(lr_mae, 2)
        },
        'bp': {
            'predictions': bp_all_pred.flatten().tolist(),
            'rmse': round(bp_rmse, 2),
            'mae': round(bp_mae, 2)
        },
        'comparison_table': [
            {'algorithm': 'Regresi Linear', 'rmse': round(lr_rmse, 2), 'mae': round(lr_mae, 2)},
            {'algorithm': 'Backpropagation (NN)', 'rmse': round(bp_rmse, 2), 'mae': round(bp_mae, 2)}
        ]
    }

    return results

# ===============================
# ROUTES
# ===============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    df = pd.read_csv(file, sep=';').dropna()
    results = train_and_compare_models(df)
    session['comparison_results'] = results
    return redirect(url_for('results'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        previous = float(request.form['previous'])
        high = float(request.form['high'])
        low = float(request.form['low'])
        actual_close = float(request.form['close'])

        # Load model & scaler
        scaler_X = joblib.load(os.path.join(MODEL_DIR, 'scaler_X.pkl'))
        scaler_y = joblib.load(os.path.join(MODEL_DIR, 'scaler_y.pkl'))
        lr_model = joblib.load(os.path.join(MODEL_DIR, 'lr_model.pkl'))
        bp_model = tf.keras.models.load_model(
            os.path.join(MODEL_DIR, 'bp_model.h5'),
            compile=False
        )

        input_data = np.array([[previous, high, low]])
        input_scaled = scaler_X.transform(input_data)

        # ===============================
        # PREDIKSI
        # ===============================
        lr_pred_scaled = lr_model.predict(input_scaled)
        lr_pred_scaled = np.array(lr_pred_scaled).reshape(-1, 1)
        lr_pred = scaler_y.inverse_transform(lr_pred_scaled)

        bp_pred_scaled = bp_model.predict(input_scaled, verbose=0)
        bp_pred = scaler_y.inverse_transform(bp_pred_scaled)

        # ===============================
        # HITUNG ERROR
        # ===============================
        rmse_lr = np.sqrt(mean_squared_error([actual_close], [lr_pred[0][0]]))
        mae_lr = mean_absolute_error([actual_close], [lr_pred[0][0]])

        rmse_bp = np.sqrt(mean_squared_error([actual_close], [bp_pred[0][0]]))
        mae_bp = mean_absolute_error([actual_close], [bp_pred[0][0]])

        # ===============================
# BUAT GRAFIK BATANG (Update: Tambah Aktual)
# ===============================
        plt.figure(figsize=(8, 5))

        # Data yang akan ditampilkan
        labels = ['Aktual', 'Regresi Linear', 'Backpropagation']
        values = [actual_close, lr_pred[0][0], bp_pred[0][0]]
        colors = ['green', 'blue', 'red'] # Hijau untuk data asli agar kontras

        bars = plt.bar(labels, values, color=colors)

        plt.ylabel('Harga Close')
        plt.title('Perbandingan Harga Aktual vs Prediksi Manual')

        # Tambahkan nilai di atas setiap batang
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2,
                    height,
                    f'{height:.2f}',
                    ha='center', 
                    va='bottom',
                    fontweight='bold')

        plt.grid(axis='y', linestyle='--', alpha=0.6)

        # Simpan ke memori untuk ditampilkan di HTML
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()


        return render_template(
            'manual_results.html',
            previous=previous,
            high=high,
            low=low,
            actual=actual_close,
            predicted_lr=round(float(lr_pred[0][0]), 2),
            predicted_bp=round(float(bp_pred[0][0]), 2),
            rmse_lr=round(rmse_lr, 4),
            mae_lr=round(mae_lr, 4),
            rmse_bp=round(rmse_bp, 4),
            mae_bp=round(mae_bp, 4),
            plot_url=plot_url
        )

    except Exception as e:
        flash(f"Error prediksi: {str(e)}")
        return redirect(url_for('index'))


@app.route('/results')
def results():
    results = session.get('comparison_results')
    if not results:
        return redirect(url_for('index'))

    # ===============================
    # 1️⃣ Grafik Actual Data
    # ===============================
    plt.figure(figsize=(10, 5))
    plt.plot(results['actual_data'], label='Actual Close')
    plt.legend()
    plt.grid(True)

    img_actual = io.BytesIO()
    plt.savefig(img_actual, format='png')
    img_actual.seek(0)
    plot_actual_url = base64.b64encode(img_actual.getvalue()).decode()
    plt.close()

    # ===============================
    # 2️⃣ Grafik Perbandingan
    # ===============================
    plt.figure(figsize=(12, 6))
    plt.plot(results['actual_data'], label='Actual')
    plt.plot(results['lr']['predictions'], label='Regresi Linear')
    plt.plot(results['bp']['predictions'], label='Backpropagation')
    plt.legend()
    plt.grid(True)

    img_compare = io.BytesIO()
    plt.savefig(img_compare, format='png')
    img_compare.seek(0)
    plot_url = base64.b64encode(img_compare.getvalue()).decode()
    plt.close()

    # ===============================
    # 3️⃣ Kirim Historical Data
    # ===============================
    historical_data = results['history']

    return render_template(
        'results.html',
        comparison_table=results['comparison_table'],
        plot_url=plot_url,
        plot_actual_url=plot_actual_url,
        historical_data=historical_data
    )


# ===============================
# MAIN
# ===============================
if __name__ == '__main__':
    app.run(debug=True)
