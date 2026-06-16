# =============================================================================
# PREDVIDJANJE PREOSTALOG ZIVOTNOG VEKA TURBOVENTILATORSKIH MLAZNIH MOTORA
# NASA C-MAPSS FD001 Dataset
# Autor: Anja Stojkovic | RA 150/2023 | SAUSAU
# =============================================================================
# Pokretanjem ovog fajla izvrsava se kompletan ML pipeline:
#   1. Eksplorativne vizualizacije originalnih podataka
#   2. Priprema i obrada podataka (ciscenje, normalizacija, feature engineering)
#   3. Baseline model (referentna tacka)
#   4. Trening XGBoost i Random Forest modela
#   5. Podesavanje hiperparametara (RandomizedSearchCV)
#   6. Odabir najznacajnijih atributa (top 10 feature selection)
#   7. Finalna evaluacija na test skupu (RMSE i NASA Score)
#   8. Vizualizacije rezultata
#   9. Zakljucak i poredjenje svih modela
# =============================================================================

from data import pripremi_sve, ucitaj_za_vizualizacije, ucitaj_test_za_vizualizacije
from vizualizacija import (vizualizuj_rul_po_motorima, vizualizuj_distribuciju_rul,
                           vizualizuj_korelaciju, vizualizuj_senzore_tokom_vremena,
                           vizualizuj_predikcije, vizualizuj_vaznost_atributa,
                           vizualizuj_poredjenje_modela, vizualizuj_rul_krivu_po_ciklusima)
from model import (baseline_model, treniraj_xgboost, treniraj_random_forest,
                   evaluiraj_na_testu, podesi_hiperpametre_xgboost,
                   treniraj_sa_najboljim_atributima)
import numpy as np
from sklearn.metrics import mean_squared_error
from model import nasa_score

# =============================================================================
# FAZA 1: EKSPLORATIVNE VIZUALIZACIJE
# Ucitavamo originalne (nenormalizovane) podatke iskljucivo za vizualizacije.
# Cilj: razumeti strukturu podataka pre bilo kakve obrade.
# =============================================================================
print("GENERISANJE VIZUALIZACIJA")
train_orig = ucitaj_za_vizualizacije()
vizualizuj_rul_po_motorima(train_orig)       # opadanje RUL kroz cikluse za 5 motora
vizualizuj_distribuciju_rul(train_orig)      # distribucija RUL vrednosti u datasetu
vizualizuj_korelaciju(train_orig)            # korelacija senzora sa RUL-om
vizualizuj_senzore_tokom_vremena(train_orig) # trend kljucnih senzora kroz cikluse

# =============================================================================
# FAZA 2: PRIPREMA PODATAKA
# Ciscenje, normalizacija, feature engineering i podela na skupove.
# Vraca 6 skupova: X i y za train, validaciju i test.
# =============================================================================
print("PRIPREMA PODATAKA")
X_train, X_val, X_test, y_train, y_val, y_test = pripremi_sve()

# =============================================================================
# FAZA 3: BASELINE MODEL
# Najjednostavniji model koji uvek predvidja prosecni RUL.
# Sluzi kao referentna tacka - pravi modeli moraju biti znacajno bolji.
# Cuvamo rmse_baseline i score_baseline za dinamicko poredjenje u grafikima.
# =============================================================================
print("BASELINE MODEL")
_, rmse_baseline, score_baseline = baseline_model(y_train, y_val, y_test)

# =============================================================================
# FAZA 4: TRENING MODELA
# Treniramo XGBoost (glavni model) i Random Forest (model za poredjenje).
# Oba modela se evaluiraju na validacionom skupu tokom treninga.
# =============================================================================
model_xgb, rmse_val_xgb, score_val_xgb = treniraj_xgboost(X_train, y_train, X_val, y_val)
model_rf, rmse_val_rf, score_val_rf = treniraj_random_forest(X_train, y_train, X_val, y_val)

# =============================================================================
# FAZA 5: PODESAVANJE HIPERPARAMETARA
# RandomizedSearchCV trazi optimalne hiperparametre za XGBoost
# kroz 20 nasumicnih kombinacija sa 3-fold cross-validacijom.
# =============================================================================
model_xgb_opt = podesi_hiperpametre_xgboost(X_train, y_train)

# =============================================================================
# FAZA 6: FINALNA EVALUACIJA NA TEST SKUPU
# Test skup nije viđen tokom treninga ni validacije - ovo je konacna ocena.
# =============================================================================
y_pred_xgb, y_pred_rf, rmse_xgb, rmse_rf, score_xgb, score_rf = evaluiraj_na_testu(
    model_xgb, model_rf, X_test, y_test)

# Evaluacija optimizovanog modela na test skupu
print("\nEVALUACIJA OPTIMIZOVANOG XGBOOST MODELA")
y_pred_opt = model_xgb_opt.predict(X_test)
rmse_opt = np.sqrt(mean_squared_error(y_test, y_pred_opt))
score_opt = nasa_score(y_test.values, y_pred_opt)
print(f"Optimizovani XGBoost -> RMSE: {rmse_opt:.2f} | NASA Score: {score_opt:.2f}")
print(f"Poboljsanje u odnosu na originalni XGBoost: {((rmse_xgb - rmse_opt) / rmse_xgb * 100):.1f}%")

# =============================================================================
# FAZA 7: ODABIR NAJZNACAJNIJIH ATRIBUTA
# Treniramo XGBoost samo sa top 10 najvaznijih atributa i poredimo sa punim modelom.
# =============================================================================
model_top10, rmse_top10, score_top10 = treniraj_sa_najboljim_atributima(
    X_train, y_train, X_val, y_val, X_test, y_test, X_train.columns.tolist())

print(f"\nPoredjenje - svi atributi vs top 10:")
print(f"XGBoost sa svim atributima: RMSE {rmse_xgb:.2f}")
print(f"XGBoost sa top 10 atributa: RMSE {rmse_top10:.2f}")

# =============================================================================
# FAZA 8: VIZUALIZACIJE REZULTATA
# =============================================================================
print("VIZUALIZACIJE REZULTATA")
test_orig = ucitaj_test_za_vizualizacije()
vizualizuj_rul_krivu_po_ciklusima(test_orig, y_pred_xgb)             # RUL kriva za testne motore
vizualizuj_predikcije(y_test, y_pred_xgb, y_pred_rf)                 # scatter: predvidjeno vs stvarno
vizualizuj_vaznost_atributa(model_xgb, model_rf, X_train.columns.tolist())  # top 15 atributa
vizualizuj_poredjenje_modela(rmse_xgb, rmse_rf, score_xgb, score_rf, rmse_baseline, score_baseline)

# =============================================================================
# FAZA 9: ZAKLJUCAK
# =============================================================================
print("\nZAKLJUCAK:")
print(f"XGBoost je {'bolji' if rmse_xgb < rmse_rf else 'losiji'} od Random Forest-a po RMSE.")
print(f"XGBoost RMSE:       {rmse_xgb:.2f} (poboljsanje od baseline-a: {((rmse_baseline - rmse_xgb) / rmse_baseline * 100):.1f}%)")
print(f"Random Forest RMSE: {rmse_rf:.2f} (poboljsanje od baseline-a: {((rmse_baseline - rmse_rf) / rmse_baseline * 100):.1f}%)")
print(f"Optimizovani XGBoost RMSE: {rmse_opt:.2f}")
print(f"XGBoost top 10 atributa RMSE: {rmse_top10:.2f}")
print(f"Odabrani model: {'XGBoost originalni' if rmse_xgb <= min(rmse_opt, rmse_top10) else 'Optimizovani XGBoost' if rmse_opt < rmse_top10 else 'XGBoost top 10'}")
