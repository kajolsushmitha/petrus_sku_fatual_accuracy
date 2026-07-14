import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Flatten, Concatenate
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(42)
np.random.seed(42)

MODEL_PATH = "sku_lstm_growth_model.keras"
WINDOW = 12
VAL_MONTHS_PER_SKU = 4


def load_data(df: pd.DataFrame) -> pd.DataFrame:
    df["Month_dt"] = pd.to_datetime(df["Month"], format="%b-%Y")
    df = df.sort_values(["SKU", "Month_dt"]).reset_index(drop=True)
    df["Accuracy"] = df["Factual Qty (Sold)"] / df["Actual Qty (Produced)"]
    df["Growth"] = df.groupby("SKU")["Factual Qty (Sold)"].pct_change()
    return df


def per_sku_growth_frames(df: pd.DataFrame, skus):
    out = {}
    for sku in skus:
        g = df[df["SKU"] == sku].sort_values("Month_dt")
        out[sku] = g.dropna(subset=["Growth"]).reset_index(drop=True)
    return out


def build_sequences(df: pd.DataFrame, skus, sku_to_id, window=WINDOW):
    per_sku = per_sku_growth_frames(df, skus)
    all_growth = np.concatenate([per_sku[s]["Growth"].values for s in skus]).reshape(-1, 1)
    growth_scaler = StandardScaler().fit(all_growth)

    X_seq, X_sku, y_out = [], [], []
    for sku in skus:
        g = per_sku[sku]
        growth_scaled = growth_scaler.transform(g["Growth"].values.reshape(-1, 1)).flatten()
        acc = g["Accuracy"].values
        months = g["Month_dt"].dt.month.values
        m_sin = np.sin(2 * np.pi * months / 12)
        m_cos = np.cos(2 * np.pi * months / 12)
        feats = np.stack([growth_scaled, acc, m_sin, m_cos], axis=1)

        for t in range(window, len(feats)):
            X_seq.append(feats[t - window:t])
            X_sku.append(sku_to_id[sku])
            y_out.append([growth_scaled[t], acc[t]])

    return np.array(X_seq), np.array(X_sku), np.array(y_out), growth_scaler, per_sku


def build_model(n_skus, window=WINDOW):
    seq_in = Input(shape=(window, 4), name="seq_in")
    sku_in = Input(shape=(1,), name="sku_in")

    emb = Embedding(input_dim=n_skus, output_dim=8)(sku_in)
    emb = Flatten()(emb)

    x = LSTM(64, return_sequences=True)(seq_in)
    x = LSTM(32)(x)

    merged = Concatenate()([x, emb])
    d = Dense(32, activation="relu")(merged)
    out = Dense(2, activation="linear")(d)

    model = Model([seq_in, sku_in], out)
    model.compile(optimizer="adam", loss="mse")
    return model


def train_model(df: pd.DataFrame, skus, sku_to_id, model_path=MODEL_PATH, verbose_label="all SKUs"):
    X_seq, X_sku, y_out, growth_scaler, per_sku = build_sequences(df, skus, sku_to_id)

    val_mask = np.zeros(len(X_sku), dtype=bool)
    for sku in skus:
        idxs = np.where(X_sku == sku_to_id[sku])[0]
        if len(idxs) > VAL_MONTHS_PER_SKU:
            val_mask[idxs[-VAL_MONTHS_PER_SKU:]] = True

    X_seq_tr, X_sku_tr, y_tr = X_seq[~val_mask], X_sku[~val_mask], y_out[~val_mask]
    X_seq_va, X_sku_va, y_va = X_seq[val_mask], X_sku[val_mask], y_out[val_mask]

    model = build_model(n_skus=len(skus))
    es = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

    print(f"Training LSTM on {len(X_seq_tr)} sequences (pooled across {verbose_label})...")
    model.fit(
        [X_seq_tr, X_sku_tr], y_tr,
        validation_data=([X_seq_va, X_sku_va], y_va) if len(X_seq_va) > 0 else None,
        epochs=200, batch_size=32, verbose=0, callbacks=[es],
    )

    if len(X_seq_va) > 0:
        preds = model.predict([X_seq_va, X_sku_va], verbose=0)
        mae_acc_pct = np.mean(np.abs(preds[:, 1] - y_va[:, 1])) * 100
        print(f"Validation accuracy-% prediction error (MAE): {mae_acc_pct:.2f} percentage points")

    model.save(model_path)
    model = load_model(model_path)
    return model, growth_scaler, per_sku


def get_model(df: pd.DataFrame, skus, sku_to_id, force_retrain=False):
    if (not force_retrain) and os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            _, _, _, growth_scaler, per_sku = build_sequences(df, skus, sku_to_id)
            return model, growth_scaler, per_sku
        except Exception as e:
            print(f"Could not load existing model, retraining... Error: {e}")
            pass
    return train_model(df, skus, sku_to_id)


def forecast_sku(model, growth_scaler, per_sku, df, sku_to_id, sku, target_month_str, window=WINDOW):
    if sku not in per_sku:
        raise ValueError(f"No data found for SKU '{sku}'.")

    g = per_sku[sku]
    last_row_full = df[df["SKU"] == sku].sort_values("Month_dt").iloc[-1]
    last_month = last_row_full["Month_dt"]
    last_level = float(last_row_full["Factual Qty (Sold)"]) 

    target_dt = pd.to_datetime(target_month_str, format="%b-%Y")
    steps = (target_dt.year - last_month.year) * 12 + (target_dt.month - last_month.month)
    if steps <= 0:
        raise ValueError(
            f"'{target_month_str}' must be AFTER the last known month "
            f"({last_month:%b-%Y}). Please pick a future month."
        )

    growth_scaled_hist = list(growth_scaler.transform(g["Growth"].values.reshape(-1, 1)).flatten())
    acc_hist = list(g["Accuracy"].values)
    months_hist = list(g["Month_dt"].dt.month.values)
    sku_id_arr = np.array([[sku_to_id[sku]]])

    cur_month = last_month
    level = last_level
    for _ in range(steps):
        cur_month = cur_month + pd.DateOffset(months=1)
        w_growth = growth_scaled_hist[-window:]
        w_acc = acc_hist[-window:]
        w_months = months_hist[-window:]
        m_sin = [np.sin(2 * np.pi * m / 12) for m in w_months]
        m_cos = [np.cos(2 * np.pi * m / 12) for m in w_months]
        feats = np.stack([w_growth, w_acc, m_sin, m_cos], axis=1)[np.newaxis, :, :]

        pred = model.predict([feats, sku_id_arr], verbose=0)[0]
        growth_real = float(growth_scaler.inverse_transform([[pred[0]]])[0][0])
        level = level * (1 + growth_real)   

        growth_scaled_hist.append(float(pred[0]))
        acc_hist.append(float(pred[1]))
        months_hist.append(cur_month.month)

    forecast_sold = max(level, 0)
    predicted_accuracy = float(np.clip(acc_hist[-1] * 100, 50, 100))
    recommended_manufacture = forecast_sold / (predicted_accuracy / 100)

    return {
        "SKU": sku,
        "Target Month": target_dt.strftime("%b-%Y"),
        "Last Known Month": last_month.strftime("%b-%Y"),
        "Factual Qty (Forecast)": round(forecast_sold, 0),
        "Predicted Accuracy %": round(predicted_accuracy, 2),
        "Recommended Manufacture Qty": round(recommended_manufacture, 0),
    }
