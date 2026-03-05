import os
import sys

# Ensure working directory is the project root (works locally and on HF Spaces)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import requests
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import gradio as gr
from datetime import datetime
from io import BytesIO
from PIL import Image

from src.train import run_future_forecast
from src.utils import load_config

SOHO_IMAGE_URL = "https://soho.nascom.nasa.gov/data/realtime/hmi_igr/512/latest.jpg"

# Load pre-saved data and config
try:
    df_filtered = joblib.load('models/sunspots_data.joblib')
    config      = joblib.load('models/config.joblib')
    validation_results = joblib.load('models/validation_results.joblib')
    print("Models i dades carregats correctament.")
except FileNotFoundError:
    print("⚠️ Fitxers de model no trobats. Carregant dades des de zero...")
    from src.data import load_data
    config      = load_config('config.yaml')
    df          = load_data(config['data']['url'], save_path=config['data']['save_path'])
    df_filtered = df.loc['1965-01-01':].copy()
    validation_results = None
    print("Dades carregades. La pestanya de validació no estarà disponible.")


def interpret_results(avg_val, last_val):
    trend    = "estable" if abs(avg_val - last_val) < 5 else ("ascendent" if avg_val > last_val else "descendent")
    activity = "alta" if avg_val > 150 else ("mitjana" if avg_val > 50 else "baixa")
    return f"Es preveu una activitat solar **{activity}** amb una tendència **{trend}** respecte a l'última observació."


def predict_sunspots(steps):
    future_predictions = run_future_forecast(df_filtered, steps=int(steps), config=config)
    last_val = df_filtered['SUNSPOTS'].iloc[-1]
    avg_val  = np.mean(list(future_predictions.values()))

    output_text = "| Data | Predicció |\n| :--- | :--- |\n"
    for date, val in future_predictions.items():
        output_text += f"| {date.strftime('%d/%m/%Y')} | {val:.2f} |\n"

    interpretation = interpret_results(avg_val, last_val)

    plt.figure(figsize=(12, 5))
    last_60   = df_filtered.tail(60)
    pred_dates = list(future_predictions.keys())
    pred_vals  = list(future_predictions.values())
    plt.plot(last_60.index, last_60['SUNSPOTS'], label='Real (Darrers 60 dies)', color='#1f77b4', linewidth=2)
    plt.plot(pred_dates, pred_vals, label='Predicció', color='#d62728', marker='o', markersize=6, linestyle='--')
    plt.title("Predicció del Nombre de Taques Solars", fontsize=14, pad=15)
    plt.xlabel("Data", fontsize=12)
    plt.ylabel("Nombre de Taques", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    return output_text, plt, interpretation


def show_validation():
    if validation_results is None:
        return None, "No hi ha dades de validació disponibles."

    plt.figure(figsize=(12, 5))
    actuals = validation_results['actuals']
    preds   = validation_results['predictions']
    show_n  = min(500, len(actuals))
    plt.plot(actuals[-show_n:], label='Real',           color='#1f77b4', alpha=0.6)
    plt.plot(preds[-show_n:],   label='Model (Hybrid)', color='#d62728', alpha=0.8, linestyle='--')
    plt.title("Rendiment Històric (Darreres 500 observacions de test)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.2)

    metrics = f"**RMSE:** {validation_results['hybrid_rmse']:.2f} | **MAE:** {validation_results['hybrid_mae']:.2f}"
    return plt, metrics


SILSO_URL = "https://www.sidc.be/silso/DATA/SN_d_tot_V2.0.csv"

def fetch_latest_silso():
    """Fetch the most recent validated row from SILSO (skips -1 missing values)."""
    from io import StringIO
    resp = requests.get(SILSO_URL, timeout=10)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text), sep=';', header=None,
                     names=['Year','Month','Day','Decimal_Date','Ri','Std','Num_obs','Definitive'])
    df = df[df['Ri'] >= 0]  # drop missing (-1) rows
    last = df.iloc[-1]
    date  = pd.Timestamp(int(last['Year']), int(last['Month']), int(last['Day']))
    count = int(last['Ri'])
    return date, count


def get_realtime_data():
    img = None
    try:
        ts   = int(pd.Timestamp.now().timestamp())
        resp = requests.get(f"{SOHO_IMAGE_URL}?t={ts}", timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))
    except Exception as e:
        print(f"Error fetching solar image: {e}")

    try:
        last_date, last_count = fetch_latest_silso()
    except Exception as e:
        print(f"Error fetching SILSO data: {e}")
        last_date  = df_filtered.index[-1]
        last_count = int(df_filtered['SUNSPOTS'].iloc[-1])

    info = (
        f"### {last_count} taques solars\n"
        f"Darrera dada validada per SILSO: **{last_date.strftime('%d/%m/%Y')}**\n\n"
        f"_Les dades de SILSO es publiquen amb 1–2 dies de retard._"
    )
    return img, info


with gr.Blocks(title="Toni's Sunspot Predictor") as demo:
    gr.Markdown("# ☀️ Taques Solars d'en Toni")
    gr.Markdown("Consulta les dades en temps real de [SILSO](https://www.sidc.be/silso/datafiles)")

    with gr.Tabs():
        with gr.Tab("🌞 Sol en Temps Real"):
            gr.Markdown("Imatge en directe del Sol (SDO/HMI Continuum, NASA) i darrera dada validada de SILSO.")
            refresh_btn = gr.Button("Actualitzar", variant="secondary")
            with gr.Row():
                with gr.Column(scale=2):
                    solar_image  = gr.Image(label="Sol ara · SDO/HMI Continuum (SOHO/NASA)", type="pil")
                with gr.Column(scale=1):
                    realtime_info = gr.Markdown()
            refresh_btn.click(get_realtime_data, outputs=[solar_image, realtime_info])
            demo.load(get_realtime_data, outputs=[solar_image, realtime_info])

        with gr.Tab("🚀 Predicció Futura"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_steps   = gr.Slider(minimum=1, maximum=30, value=7, step=1, label="Dies a predir")
                    btn           = gr.Button("Generar Predicció", variant="primary")
                    out_interpret = gr.Markdown("Fes clic per veure l'interpretació.")
                    out_table     = gr.Markdown()
                with gr.Column(scale=2):
                    out_plot = gr.Plot()
            btn.click(predict_sunspots, inputs=input_steps, outputs=[out_table, out_plot, out_interpret])

        with gr.Tab("📊 Rendiment del Model"):
            gr.Markdown("""
            ### Backtesting: predicció vs realitat
            Comparació entre les prediccions del model i els valors reals en dades que el model **mai havia vist** (validació walk-forward expandida).
            """)
            val_btn     = gr.Button("Mostrar Gràfica de Validació")
            val_metrics = gr.Markdown()
            val_plot    = gr.Plot()
            val_btn.click(show_validation, outputs=[val_plot, val_metrics])

            gr.Markdown("""
            ---
            ### Comparació amb models de referència

            | Model | Granularitat | RMSE | MAE |
            |---|---|---|---|
            | Naive (lag-1) — _demà = avui_ | diari | 22.04 | 19.59 |
            | Mitjana mòbil (30 dies) | diari | 28.47 | 26.16 |
            | McNish-Lincoln (13m smooth) — _mètode oficial SIDC_ | mensual | 32.36 | 23.64 |
            | ARIMA(5,1,0) — _model clàssic de sèries temporals_ | mensual | 24.33 | 17.51 |
            | **Model Híbrid (nostre)** | **diari** | **19.24** | **17.54** |

            El model híbrid supera el millor baseline diari (Naive lag-1) en un **−12,7% de RMSE** i un **−10,4% de MAE**.
            """)

        with gr.Tab("🌍 Per què importa?"):
            gr.Markdown("""
            ### Les taques solars i l'activitat solar

            Les taques solars són regions de la superfície del Sol on el camp magnètic és especialment intens. La seva presència indica una major activitat solar, que es manifesta en forma de **flamarades solars** i **ejeccions de massa coronal (CME)**. Quan aquestes erupcions arriben a la Terra, generen **tempestes geomagnètiques** que poden tenir conseqüències molt concretes.

            ---

            ### Per què és important predir-les?

            - **Satèl·lits i òrbita baixa**: SpaceX va perdre 38 satèl·lits Starlink el febrer de 2022 per una sola tempesta.
            - **GPS i navegació**: Les pertorbacions ionosfèriques degraden la precisió del GPS.
            - **Xarxes elèctriques**: El 1989, una tempesta va deixar tota Quebec sense llum durant 9 hores.
            - **Comunicacions HF**: Les ones d'alta freqüència es reflecteixen a la ionosfera; les pertorbacions solars poden tallar completament les comunicacions.
            - **Astronautes**: Risc directe de radiació per a l'ISS i missions a la Lluna o Mart.

            ---

            ### El cicle solar de 11 anys

            Estem en el **Cicle 25**, que va iniciar el seu mínim al desembre de 2019 i s'espera que arribi al màxim entre 2025 i 2026.
            """)

        with gr.Tab("🧠 Com funciona?"):
            gr.Markdown("""
            ### Arquitectura Híbrida

            1. **Ridge Regression**: Captura la tendència general i els cicles solars de 11 anys.
            2. **LightGBM**: Aprèn dels errors de la Ridge per modelar fluctuacions no lineals a curt termini.
            3. **Extreme Value Theory (EVT)**: Calibra els extrems — pics i caigudes sobtades que la regressió normal ignora.

            ### Validació
            Validació **walk-forward expandida**: el model mai veu el futur. Finestra inicial d'11 anys, avanç trimestral.

            **Random State 7** — el número de la sort d'en Toni.
            """)

demo.launch()
