import joblib
import pandas as pd
import gradio as gr
from datetime import datetime

# 1) Geladene Artefakte
model = joblib.load("random_forest_final.pkl")
preprocessor = joblib.load("preprocessor_final.pkl")

# 2) Defaults für numerische Features (Mittelwerte aus cars_states_cleaned.csv)
numeric_defaults = {
    "year": 2010.92,
    "condition": 31.66,
    "odometer": 60446.62,
    "sale_year": 2014.88,
    "sale_month": 3.01,
    "sale_day": 15.19,
    "sale_weekday": 1.41,
    "car_age": 3.96,
    "avg_price_state": 11730.95,
    "median_price_state": 11583.70,
    "vehicle_age": 3.97,
    "miles_per_year": 17202.40,
    "color_popularity": 3.94
}

# 3) Defaults für kategoriale Features
categorical_defaults = {
    "make": "",
    "model": "",
    "trim": "",
    "body": "",
    "transmission": "",
    "vin": "",
    "color": "white",
    "interior": "",
    "seller": "",
    "saledate": "",
    "state_code": "",
    "state_name": "",
    "state_region": "Unknown",
    "season": ""
}

# 4) Mappings & Listen
STATE_REGION = {
    "AL":"South US","AK":"West US","AZ":"West US","AR":"South US","CA":"West US",
    "CO":"West US","CT":"Northeast US","DE":"Northeast US","FL":"South US",
    "GA":"South US","HI":"West US","ID":"West US","IL":"Midwest US","IN":"Midwest US",
    "IA":"Midwest US","KS":"Midwest US","KY":"South US","LA":"South US","ME":"Northeast US",
    "MD":"Northeast US","MA":"Northeast US","MI":"Midwest US","MN":"Midwest US",
    "MS":"South US","MO":"Midwest US","MT":"West US","NE":"Midwest US","NV":"West US",
    "NH":"Northeast US","NJ":"Northeast US","NM":"West US","NY":"Northeast US",
    "NC":"South US","ND":"Midwest US","OH":"Midwest US","OK":"South US","OR":"West US",
    "PA":"Northeast US","RI":"Northeast US","SC":"South US","SD":"Midwest US",
    "TN":"South US","TX":"South US","UT":"West US","VT":"Northeast US","VA":"South US",
    "WA":"West US","WV":"South US","WI":"Midwest US","WY":"West US"
}

MONTHS_DE = ["Januar","Februar","März","April","Mai","Juni","Juli","August",
             "September","Oktober","November","Dezember"]
MONTH_MAP = {m:i+1 for i,m in enumerate(MONTHS_DE)}

COND_MAP = {"Neuwagen":5,"Vorführmodell":4,"Gebrauchtwagen":3,"Unfallfahrzeug":1}
TRIM_OPTIONS = ["Sport","Limited","LX","SE","Touring","Premium"]
GERMAN_TO_ENG_COLOR = {
    "Beige":"beige","Schwarz":"black","Blau":"blue","Braun":"brown","Bordeaux":"burgundy",
    "Anthrazit":"charcoal","Gold":"gold","Grau":"gray","Grün":"green","Limette":"lime",
    "Off-White":"off-white","Orange":"orange","Pink":"pink","Lila":"purple","Rot":"red",
    "Silber":"silver","Türkis":"turquoise","Weiß":"white","Gelb":"yellow"
}
COLOR_ENG_LIST = list(GERMAN_TO_ENG_COLOR.values())

# 5) Hilfsfunktionen
def assign_season(month:int) -> str:
    if month in (12,1,2): return "Winter"
    if month in (3,4,5):  return "Fruehling"
    if month in (6,7,8):  return "Sommer"
    return "Herbst"

def extract_flags(trim_list):
    flags = {f"has_{opt.lower()}":0 for opt in TRIM_OPTIONS}
    for t in (trim_list or []):
        key = t.lower()
        for opt in TRIM_OPTIONS:
            if opt.lower() in key:
                flags[f"has_{opt.lower()}"] = 1
    return flags

# 6) Vorhersage-Funktion mit Aufschlüsselung aller Features
def predict(
    make, state, body, color_de, month_de, year,
    km, trims, zustand
):
    # a) berechne spezifische Felder
    rec = {}
    # numerische spezifische Werte
    rec["year"] = year
    rec["condition"] = COND_MAP[zustand]
    rec["odometer"] = km * 0.621371
    sale_month = MONTH_MAP[month_de]
    rec["sale_year"] = datetime.now().year
    rec["sale_month"] = sale_month
    rec["sale_day"] = 1
    rec["sale_weekday"] = 0
    rec["car_age"] = datetime.now().year - year
    rec["avg_price_state"] = numeric_defaults["avg_price_state"]
    rec["median_price_state"] = numeric_defaults["median_price_state"]
    rec["vehicle_age"] = numeric_defaults["vehicle_age"]
    rec["miles_per_year"] = rec["odometer"] / max(rec["car_age"],1)
    eng_color = GERMAN_TO_ENG_COLOR.get(color_de, "white")
    rec["color_popularity"] = COLOR_ENG_LIST.index(eng_color)+1

    # kategoriale spezifische Werte
    rec["make"] = make
    rec["model"] = trims[0] if trims else ""
    rec["trim"] = trims[0] if trims else ""
    rec["body"] = body
    rec["transmission"] = categorical_defaults["transmission"]
    rec["vin"] = categorical_defaults["vin"]
    rec["color"] = eng_color
    rec["interior"] = categorical_defaults["interior"]
    rec["seller"] = categorical_defaults["seller"]
    rec["saledate"] = categorical_defaults["saledate"]
    rec["state_code"] = state
    rec["state_name"] = categorical_defaults["state_name"]
    rec["state_region"] = STATE_REGION.get(state, "Unknown")
    rec["season"] = assign_season(sale_month)
    # Flags
    rec.update(extract_flags(trims))

    # b) Fülle alle fehlenden Features mit Defaults
    all_feats = preprocessor.feature_names_in_
    full_rec = {}
    for feat in all_feats:
        if feat in rec:
            full_rec[feat] = rec[feat]
        elif feat in numeric_defaults:
            full_rec[feat] = numeric_defaults[feat]
        else:
            full_rec[feat] = categorical_defaults.get(feat, "Unknown")

    # c) DataFrame + Vorhersage
    df_rec = pd.DataFrame([full_rec], columns=all_feats)
    X_pre = preprocessor.transform(df_rec)
    pred = model.predict(X_pre)[0]
    return round(pred,2)

# 7) Gradio-Interface
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(["Ford","Nissan","Chevrolet","Toyota","Honda"], label="Auto-Marke"),
        gr.Dropdown(list(STATE_REGION.keys()), label="Bundesstaat (2-Buchstaben)"),
        gr.Dropdown(["Sedan","SUV","Minivan","Hatchback"], label="Karosserie"),
        gr.Dropdown(list(GERMAN_TO_ENG_COLOR.keys()), label="Farbe"),
        gr.Dropdown(MONTHS_DE, label="Kaufmonat"),
        gr.Number(label="Herstellungsjahr"),
        gr.Number(label="Kilometerstand (Zahl in km)"),
        gr.CheckboxGroup(TRIM_OPTIONS, label="Ausstattungsvarianten"),
        gr.Dropdown(list(COND_MAP.keys()), label="Zustand")
    ],
    outputs=gr.Number(label="Geschätzter Verkaufspreis"),
    title="Car Price Prediction",
    description="Füllen Sie alle Felder aus, um den geschätzten Verkaufspreis zu erhalten."
)

if __name__ == "__main__":
    demo.launch()