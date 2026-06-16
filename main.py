# Glavna skripta projekta
# Pokretanjem ovog fajla izvrsava se:
# 1. Generisanje vizualizacija
# 2. Priprema podataka za trening
# 3. Baseline model
# 4. Trening modela
# 5. Podesavanje hiperparametara
# 6. Odabir najznacajnijih atributa
# 7. Finalna evaluacija
# 8. Vizualizacije rezultata
# 9. Zakljucak

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

# VIZUALIZACIJE
print("GENERISANJE VIZUALIZACIJA")
train_orig = ucitaj_za_vizualizacije()
vizualizuj_rul_po_motorima(train_orig)
vizualizuj_distribuciju_rul(train_orig)
vizualizuj_korelaciju(train_orig)
vizualizuj_senzore_tokom_vremena(train_orig)

# PRIPREMA PODATAKA
print("PRIPREMA PODATAKA")
X_train, X_val, X_test, y_train, y_val, y_test = pripremi_sve()

# BASELINE MODEL
print("BASELINE MODEL")
_, rmse_baseline, score_baseline = baseline_model(y_train, y_val, y_test)

# TRENING MODELA
model_xgb, rmse_val_xgb, score_val_xgb = treniraj_xgboost(X_train, y_train, X_val, y_val)
model_rf, rmse_val_rf, score_val_rf = treniraj_random_forest(X_train, y_train, X_val, y_val)

# PODESAVANJE HIPERPARAMETARA
model_xgb_opt = podesi_hiperpametre_xgboost(X_train, y_train)

# FINALNA EVALUACIJA
y_pred_xgb, y_pred_rf, rmse_xgb, rmse_rf, score_xgb, score_rf = evaluiraj_na_testu(
    model_xgb, model_rf, X_test, y_test)

# EVALUACIJA OPTIMIZOVANOG MODELA
print("\nEVALUACIJA OPTIMIZOVANOG XGBOOST MODELA")
y_pred_opt = model_xgb_opt.predict(X_test)
rmse_opt = np.sqrt(mean_squared_error(y_test, y_pred_opt))
score_opt = nasa_score(y_test.values, y_pred_opt)
print(f"Optimizovani XGBoost -> RMSE: {rmse_opt:.2f} | NASA Score: {score_opt:.2f}")
print(f"Poboljsanje u odnosu na originalni XGBoost: {((rmse_xgb - rmse_opt) / rmse_xgb * 100):.1f}%")

# ODABIR NAJZNACAJNIJIH ATRIBUTA
model_top10, rmse_top10, score_top10 = treniraj_sa_najboljim_atributima(
    X_train, y_train, X_val, y_val, X_test, y_test, X_train.columns.tolist())

print(f"\nPoredjenje - svi atributi vs top 10:")
print(f"XGBoost sa svim atributima: RMSE {rmse_xgb:.2f}")
print(f"XGBoost sa top 10 atributa: RMSE {rmse_top10:.2f}")

# VIZUALIZACIJE REZULTATA
print("VIZUALIZACIJE REZULTATA")
test_orig = ucitaj_test_za_vizualizacije()
vizualizuj_rul_krivu_po_ciklusima(test_orig, y_pred_xgb)
vizualizuj_predikcije(y_test, y_pred_xgb, y_pred_rf)
vizualizuj_vaznost_atributa(model_xgb, model_rf, X_train.columns.tolist())
vizualizuj_poredjenje_modela(rmse_xgb, rmse_rf, score_xgb, score_rf, rmse_baseline, score_baseline)

# ZAKLJUCAK
print("\nZAKLJUCAK:")
print(f"XGBoost je {'bolji' if rmse_xgb < rmse_rf else 'losiji'} od Random Forest-a po RMSE.")
print(f"XGBoost RMSE:       {rmse_xgb:.2f} (poboljsanje od baseline-a: {((rmse_baseline - rmse_xgb) / rmse_baseline * 100):.1f}%)")
print(f"Random Forest RMSE: {rmse_rf:.2f} (poboljsanje od baseline-a: {((rmse_baseline - rmse_rf) / rmse_baseline * 100):.1f}%)")
print(f"Optimizovani XGBoost RMSE: {rmse_opt:.2f}")
print(f"XGBoost top 10 atributa RMSE: {rmse_top10:.2f}")
print(f"Odabrani model: {'XGBoost originalni' if rmse_xgb <= min(rmse_opt, rmse_top10) else 'Optimizovani XGBoost' if rmse_opt < rmse_top10 else 'XGBoost top 10'}")

