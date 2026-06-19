import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
import joblib


def nasa_score(y_stvarni, y_predvidjeni):

    # NASA asimetricna funkcija gubitka razvijena specificno za C-MAPSS dataset.
    # d = predvidjeno - stvarno
    #   d < 0 -> rano predvidjanje (model kaze "jos ima vremena", a motor je blize kvaru)
    #            kazna: exp(-d/13) - 1  -> blaza kazna
    #   d > 0 -> kasno predvidjanje (model kaze "motor ce duze raditi", ali ce brze otkazati)
    #            kazna: exp(d/10) - 1   -> stroza kazna


    d = y_predvidjeni - y_stvarni
    score = np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))
    return score


def baseline_model(y_train, y_val, y_test):

    # Baseline model je najjednostavniji moguci pristup: uvek predvidi prosecni RUL
    # iz trening skupa, bez obzira na vrednosti senzora.
 


    prosek = y_train.mean()
    print(f"Prosecni RUL u train skupu: {prosek:.2f}")

    y_pred_val = np.full(len(y_val), prosek)
    y_pred_test = np.full(len(y_test), prosek)

    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    score_test = nasa_score(y_test.values, y_pred_test)

    print(f"Validacioni RMSE: {rmse_val:.2f}")
    print(f"Test RMSE:        {rmse_test:.2f}")
    print(f"Test NASA Score:  {score_test:.2f}")

    return rmse_val, rmse_test, score_test


def treniraj_xgboost(X_train, y_train, X_val, y_val):

    # XGBoost (eXtreme Gradient Boosting) je ansambni model zasnovan na
    # gradijentnom pojacavanju stabala odluke.

    print("\nTRENING XGBOOST MODELA")

    model = XGBRegressor(
        n_estimators=200,    # broj stabala u ansamblu - vise stabala = preciznije, ali sporije
        max_depth=6,         # maksimalna dubina stabla - kontrolise slozenost i overfitting
        learning_rate=0.05,  # korak ucenja - manji korak = sporije ucenje, ali stabilnije
        random_state=42,     # fiksno seme za reproducibilnost rezultata
        n_jobs=-1            # koristi sva dostupna jezgra procesora za brzi trening
    )

    model.fit(X_train, y_train)

    y_pred_val = model.predict(X_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    score_val = nasa_score(y_val.values, y_pred_val)

    print(f"Validacioni RMSE: {rmse_val:.2f}")
    print(f"Validacioni NASA Score: {score_val:.2f}")

    joblib.dump(model, 'modeli/xgboost_model.pkl')
    print("Model sacuvan kao modeli/xgboost_model.pkl")

    return model, rmse_val, score_val


def treniraj_random_forest(X_train, y_train, X_val, y_val):

    # Random Forest je ansambni model koji gradi veliki broj nezavisnih stabala odluke,
    # svako na nasumicnom podskupu podataka i atributa. 

    print("\nTRENING RANDOM FOREST MODELA")

    model = RandomForestRegressor(
        n_estimators=200,  # broj stabala - vise stabala = stabilniji rezultati
        max_depth=10,      # ogranicena dubina sprjecava overfitting na trening podacima
        random_state=42,   # fiksno seme za reproducibilnost
        n_jobs=-1          # paralelni trening na svim jezgrima procesora
    )

    model.fit(X_train, y_train)

    y_pred_val = model.predict(X_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    score_val = nasa_score(y_val.values, y_pred_val)

    print(f"Validacioni RMSE: {rmse_val:.2f}")
    print(f"Validacioni NASA Score: {score_val:.2f}")

    joblib.dump(model, 'modeli/random_forest_model.pkl')
    print("Model sacuvan kao modeli/random_forest_model.pkl")

    return model, rmse_val, score_val


def evaluiraj_na_testu(model_xgb, model_rf, X_test, y_test):

    # Finalna evaluacija se vrsi iskljucivo na test skupu koji model nije video
    # tokom treninga ni validacije. 

    print("\nFINALNA EVALUACIJA NA TEST SKUPU")

    y_pred_xgb = model_xgb.predict(X_test)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    score_xgb = nasa_score(y_test.values, y_pred_xgb)

    y_pred_rf = model_rf.predict(X_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    score_rf = nasa_score(y_test.values, y_pred_rf)

    print(f"XGBoost       -> RMSE: {rmse_xgb:.2f} | NASA Score: {score_xgb:.2f}")
    print(f"Random Forest -> RMSE: {rmse_rf:.2f} | NASA Score: {score_rf:.2f}")

    return y_pred_xgb, y_pred_rf, rmse_xgb, rmse_rf, score_xgb, score_rf


def podesi_hiperpametre_xgboost(X_train, y_train):

    # Podesavanje hiperparametara metodom RandomizedSearchCV:
    # nasumicno biramo 20 kombinacija iz definisanih opsega i biramo najbolju.
    # cv=3 znaci 3-fold cross-validacija: trening skup se deli na 3 dela,
    # model se trenira na 2 i validira na 1, rotacijom - pouzdanija procena od jedne podele.
    # Parametri koje pretrazujemo:
    #   n_estimators   -> broj stabala (vise = preciznije, ali sporije)
    #   max_depth      -> dubina stabla (vece = slozenije, rizik od overfitting-a)
    #   learning_rate  -> korak ucenja (manji = stabilniji, ali treba vise stabala)
    #   subsample      -> udeo podataka za svako stablo (regularizacija)
    #   colsample_bytree -> udeo atributa za svako stablo (regularizacija)

    print("\nPODESAVANJE HIPERPARAMETARA XGBOOST")

    param_grid = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    }

    model = XGBRegressor(random_state=42, n_jobs=-1)

    search = RandomizedSearchCV(
        model,
        param_grid,
        n_iter=20,                          # broj nasumicnih kombinacija koje testiramo
        scoring='neg_root_mean_squared_error',  # optimizujemo RMSE (negativan jer sklearn maximizuje)
        cv=3,                               # 3-fold cross-validacija
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    print(f"Najbolji parametri: {search.best_params_}")
    print(f"Najbolji RMSE: {-search.best_score_:.2f}")

    joblib.dump(search.best_estimator_, 'modeli/xgboost_optimizovan.pkl')
    print("Optimizovani model sacuvan!")

    return search.best_estimator_


def treniraj_sa_najboljim_atributima(X_train, y_train, X_val, y_val, X_test, y_test, feature_names):

    # Odabir atributa smanjuje dimenzionalnost problema.
    # Koristimo feature_importances_ iz vec istreniranog XGBoost modela:
    # ovaj skor meri koliko puta je svaki atribut koriscen za podelu u stablima
    # i koliko ta podela smanjuje gresku. Vise koriscen = vazniji atribut.
    # Treniramo novi model samo sa top 10 najvaznijih atributa i poredimo performanse.
    

    print("\nODABIR NAJZNACAJNIJIH ATRIBUTA")

    model_pun = joblib.load('modeli/xgboost_model.pkl')
    importance = pd.Series(model_pun.feature_importances_, index=feature_names)
    top10 = importance.nlargest(10).index.tolist()

    print(f"Top 10 atributa: {top10}")

    X_train_top = X_train[top10]
    X_val_top = X_val[top10]
    X_test_top = X_test[top10]

    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_top, y_train)

    y_pred_val = model.predict(X_val_top)
    y_pred_test = model.predict(X_test_top)

    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    score_test = nasa_score(y_test.values, y_pred_test)

    print(f"Model sa top 10 atributa:")
    print(f"Validacioni RMSE: {rmse_val:.2f}")
    print(f"Test RMSE:        {rmse_test:.2f}")
    print(f"Test NASA Score:  {score_test:.2f}")

    joblib.dump(model, 'modeli/xgboost_top10.pkl')

    return model, rmse_test, score_test
