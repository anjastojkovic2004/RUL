import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def vizualizuj_rul_po_motorima(train):

    # Prikazuje kako RUL linearno opada kroz cikluse za prvih 5 motora.
    # Svaki motor pocinje sa razlicitim maksimalnim RUL-om (jer je cap na 125,
    # ali motori imaju razlicit broj ciklusa do kvara).
    # opadajuce linije koje zavrsavaju na RUL=0 (trenutak kvara).

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

    # Prikazuje distribuciju RUL vrednosti u celom trening skupu.


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

    # Prikazuje Pearsonovu korelaciju svakog senzora sa RUL vrednoscu.
    # Korelacija blizu -1: senzor raste kako motor stari (indikator degradacije)
    # Korelacija blizu +1: senzor opada kako motor stari
    # Korelacija blizu  0: senzor ne nosi informaciju o degradaciji
    # Crvene trake -> negativna korelacija (senzori koji rastu pri degradaciji)
    # Plave trake  -> pozitivna korelacija (senzori koji opadaju pri degradaciji)
 

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

    # Prikazuje kako se 4 kljucna senzora menjaju kroz cikluse.
    # s11, s4 (crveni) -> negativna korelacija: vrednosti rastu kako se motor kvari
    #                     fizicki: temperatura/pritisak raste pri degradaciji kompresora
    # s12, s7 (plavi)  -> pozitivna korelacija: vrednosti opadaju kako se motor kvari
    #                     fizicki: efikasnost pada pri degradaciji


    motor1 = train[train['unit'] == 1]

    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Promena najvaznijih senzora kroz cikluse (Motor 1)', fontsize=14)

    axs[0, 0].plot(motor1['ciklus'], motor1['s11'], color='red')
    axs[0, 0].set_title('Senzor s11 (negativna korelacija)')
    axs[0, 0].set_xlabel('Ciklus')
    axs[0, 0].set_ylabel('Vrednost')
    axs[0, 0].grid(True)

    axs[0, 1].plot(motor1['ciklus'], motor1['s4'], color='red')
    axs[0, 1].set_title('Senzor s4 (negativna korelacija)')
    axs[0, 1].set_xlabel('Ciklus')
    axs[0, 1].set_ylabel('Vrednost')
    axs[0, 1].grid(True)

    axs[1, 0].plot(motor1['ciklus'], motor1['s12'], color='steelblue')
    axs[1, 0].set_title('Senzor s12 (pozitivna korelacija)')
    axs[1, 0].set_xlabel('Ciklus')
    axs[1, 0].set_ylabel('Vrednost')
    axs[1, 0].grid(True)

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

    # Prikazuje predvidjeni RUL u kontekstu celokupne degradacione krive motora.
    # Test skup sadrzi prekinute serije ne znamo tacno koliko je motor radio pre testa.


    if motor_ids is None:
        motor_ids = [1, 2, 3, 4]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Stvarni vs Predvidjeni RUL kroz cikluse (testni motori)', fontsize=14)
    axs = axs.flatten()

    for i, motor_id in enumerate(motor_ids):
        motor = test_orig[test_orig['unit'] == motor_id].copy()
        n = len(motor)

        # Rekonstrukcija krive: od (n + finalni_rul) do finalni_rul, opadajuci za 1 po ciklusu
        finalni_rul = y_pred_xgb[motor_id - 1]
        stvarni_rul = list(range(n + int(finalni_rul), int(finalni_rul), -1))

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
    # Scatter plot: svaka tacka = jedan testni motor.
    # X-osa = stvarni RUL (iz RUL_FD001.txt), Y-osa = predvidjeni RUL modela.
    # Idealna predikcija: sve tacke leze na crvenoj dijagonali (predvidjeno = stvarno).
    # Tacke iznad dijagonale -> model precenjuje RUL (kasno predvidjanje, opasnije)
    # Tacke ispod dijagonale -> model podcenjuje RUL (rano predvidjanje, konzervativno)
    # Gustina tacaka oko dijagonale govori o preciznosti modela.
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Predvidjene vs Stvarne RUL vrednosti', fontsize=14)

    axs[0].scatter(y_test, y_pred_xgb, alpha=0.5, color='steelblue', s=20)
    axs[0].plot([0, 125], [0, 125], 'r--', linewidth=2, label='Idealna linija')
    axs[0].set_xlabel('Stvarni RUL')
    axs[0].set_ylabel('Predvidjeni RUL')
    axs[0].set_title('XGBoost')
    axs[0].legend()
    axs[0].grid(True)

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

    # Prikazuje top 15 najvaznijih atributa za svaki model.
    # Vaznost (feature importance) meri koliko svaki atribut doprinosi
    # smanjenju greske predikcije kroz sva stabla u ansamblu.
    # Atributi sa _roll5 sufiksom su pokretni proseci ako su visoko rangirani,
    # potvrdjuje se da glajcanje suma poboljsava predikciju.


    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Vaznost atributa', fontsize=14)

    importance_xgb = pd.Series(model_xgb.feature_importances_, index=feature_names)
    importance_xgb.nlargest(15).sort_values().plot(kind='barh', ax=axs[0], color='steelblue')
    axs[0].set_title('XGBoost - Top 15 atributa')
    axs[0].set_xlabel('Vaznost')
    axs[0].grid(True)

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

    # Uporedni bar chart RMSE i NASA Score za sva tri modela: Baseline, XGBoost, Random Forest.
    # RMSE grafik: manji stub = preciznije predvidjanje u proseku
    # NASA Score grafik: manji stub = manje kumulativne kazne (posebno za kasna predvidjanja)
    # Ocekivano: XGBoost i Random Forest daleko ispod baseline-a po oba kriterijuma.
    # Baseline vrednosti su dinamicki izracunate (nisu hardkodovane) da grafik uvek
    # prikazuje tacne vrednosti bez obzira na promene u podacima.
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Poredjenje modela', fontsize=14)

    modeli = ['Baseline', 'XGBoost', 'Random Forest']
    rmse_vrednosti = [rmse_baseline, rmse_xgb, rmse_rf]
    score_vrednosti = [score_baseline, score_xgb, score_rf]
    boje = ['red', 'steelblue', 'green']

    axs[0].bar(modeli, rmse_vrednosti, color=boje)
    axs[0].set_title('RMSE (manji = bolji)')
    axs[0].set_ylabel('RMSE')
    axs[0].grid(True, axis='y')
    for i, v in enumerate(rmse_vrednosti):
        axs[0].text(i, v + 0.5, f'{v:.2f}', ha='center', fontweight='bold')

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
