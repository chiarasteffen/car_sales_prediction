# Car Sales Price Prediction

## Project Description
Predicts the sales price of used cars based on make, model, year, condition, mileage and regional factors.

## Results
*Ergebnisse hier eintragen, z. B. Performance der finalen Modelle und Interpretation.*

### Name & URL
| Name         | URL |
|--------------|-----|
| Huggingface  | [Space URL](https://huggingface.co/spaces/USERNAME/car-price-prediction) |
| Code         | [GitHub Repository](https://github.com/USERNAME/car-price-prediction) |

## Data Sources and Features Used Per Source
| Data Source      | Features                                                                 |
|------------------|--------------------------------------------------------------------------|
| `car_prices.csv` | vin, make, model, trim, year, condition, odometer, mmr, sellingprice, saledate, transmission, color, interior, seller, state |
| `states.csv`     | State Code, Region, Division                                             |

## Features Created
| Feature        | Description                                                         |
|----------------|---------------------------------------------------------------------|
| `sale_year`    | Jahr des Verkaufs aus `saledate`                                    |
| `sale_month`   | Monat des Verkaufs aus `saledate`                                   |
| `sale_weekday` | Wochentag des Verkaufs als Integer (0=Montag)                       |
| `car_age`      | Differenz aus `sale_year` und Herstellungsjahr `year`               |
| `price_to_mmr` | Verhältnis `sellingprice / mmr`                                      |
| `odometer_cat` | Kategorisierung des Kilometerstands (z. B. 0–50k, 50k–100k, >100k)   |
| `region`       | Region des Verkäufers aus `states.csv`                              |
| `division`     | Division des Verkäufers aus `states.csv`                            |
| `make_encoded` | Target- oder One-Hot-Encoding für `make`                             |
| `model_encoded`| Target- oder One-Hot-Encoding für `model`                            |

## Model Training
### Amount of Data
- Gesamtzahl Datensätze: 558 837

### Data Splitting Method (Train/Validation/Test)
- z. B. 80/20 Split, anschließend 5-Fold Cross-Validation auf Trainingsdaten

### Performance
| It. Nr | Modell             | Metriken (MAE, RMSE, R²)       | Features                  | Beschreibung              |
|--------|--------------------|--------------------------------|---------------------------|---------------------------|
| 1      | DummyRegressor     | MAE=…, RMSE=…, R²=…            | Baseline                  | Referenzmodell            |
| 2      | Linear Regression  | MAE=…, RMSE=…, R²=…            | Numerical + Region        | Underfitting/Overfitting? |
| 3      | Random Forest      | MAE=…, RMSE=…, R²=…            | + Feature Engineering     | Tuning nötig              |
| 4      | LightGBM           | MAE=…, RMSE=…, R²=…            | Optimized                 | Finale Auswahl            |

## References
- Feature Importance und weitere Plots im Verzeichnis `doc/`
