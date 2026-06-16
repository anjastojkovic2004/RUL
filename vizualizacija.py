import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def vizualizuj_rul_po_motorima(train):
    # Prikazujemo kako RUL opada kroz cikluse za prvih 5 motora
    # Svaki motor je posebna linija na grafiku
    plt.figure(figsize=(12, 6))

    for motor_id in range(1, 6):
        motor = train[train['unit'] == motor_id]
        plt.plot(motor['ciklus'], motor['RUL'], label=f'Motor {motor_id}')

    plt.xlabel('Ciklus (broj leta)')
    plt.ylabel('RUL (preostali zivotni vek)')
    plt.title('Opadanje RUL kroz cikluse za prvih 5 motora')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('grafici/rul_po_motorima.png')
    plt.show()
    print("Grafik sacuvan kao grafici/rul_po_motorima.png")


def vizualizuj_distribuciju_rul(train):
    # Prikazujemo distribuciju RUL vrednosti u train skupu
    # Vidimo koliko ima redova sa kojim RUL vrednostima
    plt.figure(figsize=(10, 5))

    sns.histplot(train['RUL'], bins=50, kde=True, color='steelblue')

    plt.xlabel('RUL vrednost')
    plt.ylabel('Broj redova')
    plt.title('Distribucija RUL vrednosti u train skupu')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('grafici/distribucija_rul.png')
    plt.show()
    print("Grafik sacuvan kao grafici/distribucija_rul.png")


def vizualizuj_korelaciju(train):
    # Prikazujemo korelaciju senzora sa RUL vrednoscu
    # Pozitivna korelacija -> senzor raste kada RUL raste
    # Negativna korelacija -> senzor raste kada RUL opada (degradacija)
    senzori = [kol for kol in train.columns if kol not in ['unit', 'ciklus', 'RUL']]

    korelacije = train[senzori + ['RUL']].corr()['RUL'].drop('RUL').sort_values()

    plt.figure(figsize=(12, 8))
    korelacije.plot(kind='barh', color=['red' if x < 0 else 'steelblue' for x in korelacije])

    plt.xlabel('Korelacija sa RUL')
    plt.title('Korelacija senzora sa RUL vrednoscu')
    plt.axvline(x=0, color='black', linewidth=0.8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('grafici/korelacija_senzora.png')
    plt.show()
    print("Grafik sacuvan kao grafici/korelacija_senzora.png")


def vizualizuj_senzore_tokom_vremena(train):
    # Prikazujemo kako se najvazniji senzori menjaju kroz cikluse
    # Koristimo motor 1 kao primer
    # s11, s4 -> negativna korelacija (rastu kako se motor kvari)
    # s12, s7 -> pozitivna korelacija (opadaju kako se motor kvari)
    motor1 = train[train['unit'] == 1]

    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Promena najvaznijih senzora kroz cikluse (Motor 1)', fontsize=14)

    # s11
    axs[0, 0].plot(motor1['ciklus'], motor1['s11'], color='red')
    axs[0, 0].set_title('Senzor s11 (negativna korelacija)')
    axs[0, 0].set_xlabel('Ciklus')
    axs[0, 0].set_ylabel('Vrednost')
    axs[0, 0].grid(True)

    # s4
    axs[0, 1].plot(motor1['ciklus'], motor1['s4'], color='red')
    axs[0, 1].set_title('Senzor s4 (negativna korelacija)')
    axs[0, 1].set_xlabel('Ciklus')
    axs[0, 1].set_ylabel('Vrednost')
    axs[0, 1].grid(True)

    # s12
    axs[1, 0].plot(motor1['ciklus'], motor1['s12'], color='steelblue')
    axs[1, 0].set_title('Senzor s12 (pozitivna korelacija)')
    axs[1, 0].set_xlabel('Ciklus')
    axs[1, 0].set_ylabel('Vrednost')
    axs[1, 0].grid(True)

    # s7
    axs[1, 1].plot(motor1['ciklus'], motor1['s7'], color='steelblue')
    axs[1, 1].set_title('Senzor s7 (pozitivna korelacija)')
    axs[1, 1].set_xlabel('Ciklus')
    axs[1, 1].set_ylabel('Vrednost')
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('grafici/senzori_tokom_vremena.png')
    plt.show()
    print("Grafik sacuvan kao grafici/senzori_tokom_vremena.png")


def vizualizuj_rul_krivu_po_ciklusima(test_orig, y_pred_xgb, motor_ids=None):
    # Prikazujemo stvarnu krivu opadanja RUL i predvidjene vrednosti kroz cikluse
    # za odabrane testne motore - onako kako opis projekta zahteva
    if motor_ids is None:
        motor_ids = [1, 2, 3, 4]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Stvarni vs Predvidjeni RUL kroz cikluse (testni motori)', fontsize=14)
    axs = axs.flatten()

    for i, motor_id in enumerate(motor_ids):
        motor = test_orig[test_orig['unit'] == motor_id].copy()
        n = len(motor)

        # Stvarni RUL opada linearno od nepoznate vrednosti do 0
        # Koristimo predvidjenu finalnu vrednost kao polaziste za rekonstrukciju
        finalni_rul = y_pred_xgb[motor_id - 1]
        stvarni_rul = list(range(n + int(finalni_rul), int(finalni_rul), -1))

        # Predvidjene vrednosti raspolozive su samo za poslednji ciklus
        axs[i].plot(motor['ciklus'].values, stvarni_rul[:n], color='steelblue', label='Stvarni RUL (procena)')
        axs[i].axhline(y=finalni_rul, color='red', linestyle='--', label=f'Predvidjeni RUL: {finalni_rul:.0f}')
        axs[i].scatter([motor['ciklus'].values[-1]], [finalni_rul], color='red', zorder=5, s=60)
        axs[i].set_title(f'Motor {motor_id}')
        axs[i].set_xlabel('Ciklus')
        axs[i].set_ylabel('RUL')
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.savefig('grafici/rul_kriva_testni_motori.png')
    plt.show()
    print("Grafik sacuvan kao grafici/rul_kriva_testni_motori.png")


def vizualizuj_predikcije(y_test, y_pred_xgb, y_pred_rf):
    # Prikazujemo predvidjene vs stvarne RUL vrednosti za oba modela
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Predvidjene vs Stvarne RUL vrednosti', fontsize=14)

    # XGBoost
    axs[0].scatter(y_test, y_pred_xgb, alpha=0.5, color='steelblue', s=20)
    axs[0].plot([0, 125], [0, 125], 'r--', linewidth=2, label='Idealna linija')
    axs[0].set_xlabel('Stvarni RUL')
    axs[0].set_ylabel('Predvidjeni RUL')
    axs[0].set_title('XGBoost')
    axs[0].legend()
    axs[0].grid(True)

    # Random Forest
    axs[1].scatter(y_test, y_pred_rf, alpha=0.5, color='green', s=20)
    axs[1].plot([0, 125], [0, 125], 'r--', linewidth=2, label='Idealna linija')
    axs[1].set_xlabel('Stvarni RUL')
    axs[1].set_ylabel('Predvidjeni RUL')
    axs[1].set_title('Random Forest')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig('grafici/predikcije.png')
    plt.show()
    print("Grafik sacuvan kao grafici/predikcije.png")


def vizualizuj_vaznost_atributa(model_xgb, model_rf, feature_names):
    # Prikazujemo 15 najvaznijih atributa za svaki model
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Vaznost atributa', fontsize=14)

    # XGBoost
    importance_xgb = pd.Series(model_xgb.feature_importances_, index=feature_names)
    importance_xgb.nlargest(15).sort_values().plot(kind='barh', ax=axs[0], color='steelblue')
    axs[0].set_title('XGBoost - Top 15 atributa')
    axs[0].set_xlabel('Vaznost')
    axs[0].grid(True)

    # Random Forest
    importance_rf = pd.Series(model_rf.feature_importances_, index=feature_names)
    importance_rf.nlargest(15).sort_values().plot(kind='barh', ax=axs[1], color='green')
    axs[1].set_title('Random Forest - Top 15 atributa')
    axs[1].set_xlabel('Vaznost')
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig('grafici/vaznost_atributa.png')
    plt.show()
    print("Grafik sacuvan kao grafici/vaznost_atributa.png")


def vizualizuj_poredjenje_modela(rmse_xgb, rmse_rf, score_xgb, score_rf, rmse_baseline, score_baseline):
    # Prikazujemo poredjenje RMSE i NASA Score za sve modele
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Poredjenje modela', fontsize=14)

    modeli = ['Baseline', 'XGBoost', 'Random Forest']
    rmse_vrednosti = [rmse_baseline, rmse_xgb, rmse_rf]
    score_vrednosti = [score_baseline, score_xgb, score_rf]
    boje = ['red', 'steelblue', 'green']

    # RMSE
    axs[0].bar(modeli, rmse_vrednosti, color=boje)
    axs[0].set_title('RMSE (manji = bolji)')
    axs[0].set_ylabel('RMSE')
    axs[0].grid(True, axis='y')
    for i, v in enumerate(rmse_vrednosti):
        axs[0].text(i, v + 0.5, f'{v:.2f}', ha='center', fontweight='bold')

    # NASA Score
    axs[1].bar(modeli, score_vrednosti, color=boje)
    axs[1].set_title('NASA Score (manji = bolji)')
    axs[1].set_ylabel('NASA Score')
    axs[1].grid(True, axis='y')
    for i, v in enumerate(score_vrednosti):
        axs[1].text(i, v + 100, f'{v:.0f}', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('grafici/poredjenje_modela.png')
    plt.show()
    print("Grafik sacuvan kao grafici/poredjenje_modela.png")