# %%
import joblib
import pandas as pd
import numpy as np
import gradio as gr
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# --- 1) Model & Data laden ---
MODEL_PATH = "models/random_forest_final.pkl"
DATA_PATH = "data/processed/cars_states_cleaned.csv"

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# --- 2) Default-Werte und State-Info ---
default_trim = df["trim"].mode()[0]
default_interior = df["interior"].mode()[0]

state_info = (
    df
    .groupby("state_clean")
    .agg(
        state_region=("state_region","first"),
        avg_price_state=("avg_price_state","first"),
        median_price_state=("median_price_state","first")
    )
    .to_dict(orient="index")
)

# --- 3) Pipeline rekonstruieren ---
numeric_features = [
    "condition","odometer","year","sale_year","sale_month","sale_day",
    "sale_weekday","car_age","avg_price_state","median_price_state",
    "miles_per_year","color_popularity"
]
categorical_features = [
    "state_region","body","make","model","trim","color","interior","season"
]  # Flags über trim

num_pipe = Pipeline([("imputer", SimpleImputer("median")),("scaler", StandardScaler())])
cat_pipe = Pipeline([("imputer", SimpleImputer("most_frequent")),("onehot", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer([("num", num_pipe, numeric_features),("cat", cat_pipe, categorical_features)])

# --- 4) Hilfsfunktionen ---
def assign_season(month: int) -> str:
    if month in (12,1,2): return "Winter"
    if month in (3,4,5):  return "Fruehling"
    if month in (6,7,8):  return "Sommer"
    return "Herbst"

def extract_flags(trim: str):
    """Leitet Flags aus dem trim-String ab."""
    kw = trim.lower()
    return {
        "has_sport": int("sport" in kw),
        "has_limited": int("limited" in kw),
        "has_lx": int("lx" in kw),
        "has_se": int("se" in kw and "se" not in ["season"]), 
        "has_touring": int("touring" in kw),
        "has_premium": int("premium" in kw)
    }

# --- 5) Vorhersage-Funktion ---
def predict(
    make, state, body, color, month, year,
    condition, odometer, trim, interior
):
    # Berechnete Felder
    season = assign_season(month)
    current_year = datetime.now().year
    age = current_year - year
    sale_date = datetime(current_year, month, 1)
    sale_year, sale_month, sale_day, sale_weekday = sale_date.year, sale_date.month, sale_date.day, sale_date.weekday()
    usage = odometer / max(age,1)
    color_rank = int(df["color"].value_counts().rank(method="dense",ascending=False).get(color,1))
    flags = extract_flags(trim)
    # State-spezifische Werte
    info = state_info[state]
    region = info["state_region"]
    avg_price = info["avg_price_state"]
    med_price = info["median_price_state"]
    # Input-Dict
    rec = {
        "condition": condition,
        "odometer": odometer,
        "year": year,
        "sale_year": sale_year,
        "sale_month": sale_month,
        "sale_day": sale_day,
        "sale_weekday": sale_weekday,
        "car_age": age,
        "avg_price_state": avg_price,
        "median_price_state": med_price,
        "miles_per_year": usage,
        "color_popularity": color_rank,
        "state_region": region,
        "body": body,
        "make": make,
        "model": default_trim,
        "trim": trim,
        "color": color,
        "interior": interior,
        "season": season,
        **flags
    }
    df_rec = pd.DataFrame([rec])
    X_pre = preprocessor.transform(df_rec)
    pred = model.predict(X_pre)[0]
    return round(pred,2)

# --- 6) Gradio-Interface ---
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(["Ford","Nissan","Chevrolet","Toyota","Honda"], label="Make"),
        gr.Dropdown(list(state_info.keys()), label="State"),
        gr.Dropdown(["Sedan","SUV","Minivan","Hatchback"], label="Body"),
        gr.Dropdown(df["color"].unique().tolist(), label="Color"),
        gr.Slider(1,12, value=6, step=1, label="Purchase Month"),
        gr.Slider(2000, datetime.now().year, value=2015, step=1, label="Year"),
        gr.Slider(1,5, value=3, step=1, label="Condition (1–5)"),
        gr.Number(default=df["odometer"].median(), label="Odometer"),
        gr.Textbox(default=default_trim, label="Trim"),
        gr.Textbox(default=default_interior, label="Interior")
    ],
    outputs=gr.Number(label="Predicted Selling Price"),
    title="Car Price Prediction",
    description="Fülle alle Felder aus, um den geschätzten Verkaufspreis zu erhalten."
)

if __name__ == "__main__":
    demo.launch()
