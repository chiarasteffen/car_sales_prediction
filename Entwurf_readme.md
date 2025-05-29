# Car Sales Price Prediction

## Project Description
Prognostiziert den Verkaufspreis von Gebrauchtwagen anhand von Marke, Modell, Baujahr, Zustand, Kilometerstand und regionalen Faktoren.

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
| [Kaggle](https://www.kaggle.com/datasets/zaynalabidin/car-sales-prices) | year, make, model, trim, body, transmission, vin, state, condition, odometer, color, interior, seller, mmr, sellingprice, saledate |
| [Kaggle](https://www.kaggle.com/datasets/akk26001/us-and-canada-states)     | StateCode, StateName, Region, AlternateName |

## Features Created
| Feature | Description  | Creation  |
|---------|-------------|-------------|
| **sale_year**, **sale_month**, **sale_day**, **sale_weekday** | Zeitkomponenten des Verkaufsdatums | Extraktion aus `saledate` via `.dt.year/.month/.day/.weekday`|
| **state_clean**, **state_code**, **state_name**, **state_region** | Bereinigte und gematchte Staatsinformationen | Strip und Uppercase, dann Merge mit `states`-Lookup-Tabelle |
| **car_age** | Alter des Fahrzeugs | `sale_year` − `year`  |
| **avg_price_state** | Durchschnittlicher Verkaufspreis pro Bundesstaat| Gruppenmittel aller `sellingprice` nach `state_code` |
| **median_price_state** | Medianer Verkaufspreis pro Bundesstaat | Gruppenmedian aller `sellingprice` nach `state_code` |
| **season** | Jahreszeit des Verkaufs | Mapping `sale_month` → {Winter, Fruehling, Sommer, Herbst} |
| **mean_price_make_model_season** | Durchschnittspreis je Marke‐Modell‐Jahreszeit | Gruppenmittel von `sellingprice` für jede (`make`, `model`, `season`)-Kombination |
| **has_sport**, **has_limited**, **has_lx**, **has_se**, **has_touring**, **has_premium** | Stichwort‐Flags aus Trim/Model | `.str.contains(kw)` auf Spalten `trim` und `model` für je ein Keyword |
| **vehicle_age** | Fahrzeugalter (Vorstufe für Jahresfahrleistung) | `sale_year` − `year` mit 0→1 Korrektur |
| **miles_per_year** | Durchschnittliche Jahresfahrleistung | `odometer` ÷ `vehicle_age` |
| **color_popularity** | Rang der Lackfarbe nach Häufigkeit | Dichte‐Rang der Werte in `color` (1 = häufigste Farbe) |


## Model Training
### Amount of Data
- Gesamtzahl Datensätze: 98'129 Autos

### Data Splitting Method (Train/Validation/Test)
- z. B. 80/20 Split, anschliessend 5-Fold Cross-Validation auf Trainingsdaten

### Performance
| It. Nr | Model                                | Performance                                                                                              | Features                                           | Description                                            |
|--------|--------------------------------------|----------------------------------------------------------------------------------------------------------|----------------------------------------------------|--------------------------------------------------------|
| 1      | Linear Regression (mit MMR)          | Train → MAE: 887.85, RMSE: 1301.62, R²: 0.9318<br>Test → MAE: 893.45, RMSE: 1306.29, R²: 0.9322            | Alle originalen Features inkl. `mmr`                | Baseline-Performance mit MMR                           |
| 2      | Random Forest (mit MMR)              | Train → MAE: 288.16, RMSE: 423.44, R²: 0.9928<br>Test → MAE: 778.56, RMSE: 1141.82, R²: 0.9482            | Alle originalen Features inkl. `mmr`                | Starker Overfitting durch MMR-Leak                     |
| 3      | Random Forest (ohne MMR)             | Train → MAE: 671.21, RMSE: 912.90, R²: 0.9659<br>Test → MAE: 1804.50, RMSE: 2441.23, R²: 0.7567           | Ohne `mmr`, Ausreißer gefiltert, `median_price_state` entfernt | Baseline nach MMR-Entfernung                           |
| 4      | Random Forest (getunt)               | Train → MAE: 1126.92, RMSE: 1538.00, R²: 0.9033<br>Test → MAE: 1781.53, RMSE: 2405.00, R²: 0.7639          | Wie It. 3 + Hyperparameter-Tuning                   | Overfitting reduziert, leichte Test-Verbesserung       |
| 5      | GradientBoostingRegressor            | CV RMSE: 2539.35 ± 17.85<br>Train → MAE: 1913.80, RMSE: 2527.95, R²: 0.7387<br>Test → MAE: 1913.43, RMSE: 2531.92, R²: 0.7383 | Wie It. 3                                           | Alternative Modellklasse, geringere Genauigkeit        |
| 6      | Random Forest (erweiterte Features)  | Train → MAE: 1096.89, RMSE: 1498.45, R²: 0.9082<br>Test → MAE: 1788.88, RMSE: 2414.38, R²: 0.7620          | Wie It. 4 + neue Features (age, season, flags, cyclical, bins) | Neue Features bringen keinen Test-Gewinn               |
| 7      | Linear Regression (final)            | Test → MAE: 1369.15, RMSE: 1874.48, R²: 0.8605                                                            | Alle Features aus Notebook 1 ohne `mmr`             | Endgültige Evaluation der Linearen Regression         |
| 8      | Random Forest (final)                | Test → MAE: 921.82, RMSE: 1332.17, R²: 0.9295                                                            | Alle Features aus Notebook 1 ohne `mmr`             | Endgültige Evaluation – Random Forest als Bestes Modell |


## References
- Feature Importance und weitere Plots im Verzeichnis `doc/`
