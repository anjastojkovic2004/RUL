import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib


def pripremi_sve():

    # UCITAVANJE PODATAKA
    # .txt datoteka nema zaglavlja pa moramo da ih definisemo
    # unit -> id motora
    # ciklus -> redni broj leta
    # op1.. -> operativna podesavanja (visina, brzina, ugao)
    # s1.. -> merenje sa 21 senzora
    kolone = ['unit', 'ciklus', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]

    train = pd.read_csv('data/train_FD001.txt', sep=r'\s+', header=None, names=kolone)
    test = pd.read_csv('data/test_FD001.txt', sep=r'\s+', header=None, names=kolone)
    rul = pd.read_csv('data/RUL_FD001.txt', header=None, names=['RUL'])

    # train -> 20631 redova i 26 kolona
    # test -> 13096 redova i 26 kolona
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    print("\nPrvih 5 redova train podataka:")
    print(train.head())
    print("\nPrvih 5 redova test podataka:")
    print(test.head())

    # PROVERA NEDOSTAJUCIH VREDNOSTI
    # Proveravamo da li u podacima ima praznih polja (NaN vrednosti)
    # Ako ima, moramo ih popuniti ili ukloniti pre treninga
    nan_train = train.isnull().sum().sum()
    nan_test = test.isnull().sum().sum()

    print(f"\nBroj NaN vrednosti u train skupu: {nan_train}")
    print(f"Broj NaN vrednosti u test skupu: {nan_test}")

    if nan_train == 0 and nan_test == 0:
        print("Nema nedostajucih vrednosti - podaci su kompletni!")
    else:
        print("Postoje nedostajuce vrednosti - potrebno ciscenje!")

    # IZRACUNAVANJE RUL ZA TRENING SKUP
    # Za svaki motor pronalazimo maksimalni broj ciklusa (poslednji let/trenutak kvara).
    # Zatim racunamo RUL po formuli: RUL = maksimalni ciklus - trenutni ciklus
    max_ciklus = train.groupby('unit')['ciklus'].max().reset_index()
    max_ciklus.columns = ['unit', 'max_ciklus']
    train = train.merge(max_ciklus, on='unit')
    train['RUL'] = train['max_ciklus'] - train['ciklus']
    train.drop(columns=['max_ciklus'], inplace=True)

    print("\nTrain sa RUL kolonom:")
    print(train[['unit', 'ciklus', 'RUL']].head(10))
    print("\nMaksimalni RUL:", train['RUL'].max())
    print("Minimalni RUL:", train['RUL'].min())

    # UKLANJANJE KONSTANTNIH SENZORA
    # Pronalazimo sve senzore cija je standardna devijacija jednaka nuli (ne menjaju se tokom vremena)
    # i uklanjamo ih iz oba skupa podataka. To su: op3, s1, s10, s18 i s19
    konstantni = [kol for kol in train.columns if train[kol].std() == 0]
    print("\nKonstantni senzori (uklanjamo ih):", konstantni)
    train.drop(columns=konstantni, inplace=True)
    test.drop(columns=konstantni, inplace=True)

    print("Train shape posle uklanjanja:", train.shape)
    print("Test shape posle uklanjanja:", test.shape)

    # NORMALIZACIJA
    # Koristimo MinMaxScaler koji sve vrednosti senzora prevodi na opseg od 0 do 1.
    # Scaler treniramo iskljucivo na train podacima, a iste parametre primenjujemo na test skup
    senzori = [kol for kol in train.columns if kol not in ['unit', 'ciklus', 'RUL']]
    scaler = MinMaxScaler()
    train[senzori] = scaler.fit_transform(train[senzori])
    test[senzori] = scaler.transform(test[senzori])

    print("\nPodaci posle normalizacije (min/max trebaju biti 0/1):")
    print(train[senzori].describe().loc[['min', 'max']].round(2))

    # OGRANICAVANJE RUL NA 125
    # Sve RUL vrednosti vece od 125 postavljamo na 125 koristeci clip(upper=125).
    RUL_MAX = 125
    train['RUL'] = train['RUL'].clip(upper=RUL_MAX)

    print("\nRUL posle ogranicavanja:")
    print("Maksimalni RUL:", train['RUL'].max())
    print("Minimalni RUL:", train['RUL'].min())

    # FEATURE ENGINEERING - POKRETNI PROSECI
    # Za svaki senzor dodajemo novu kolonu koja predstavlja pokretni prosek poslednjih 5 ciklusa
    # (rolling average sa prozorom 5). Na primer, za senzor s2 dodajemo kolonu s2_roll5.
    senzori_bez_id = [kol for kol in train.columns if kol not in ['unit', 'ciklus', 'RUL']]
    for kol in senzori_bez_id:
        train[f'{kol}_roll5'] = train.groupby('unit')[kol].transform(lambda x: x.rolling(5, min_periods=1).mean())
        test[f'{kol}_roll5'] = test.groupby('unit')[kol].transform(lambda x: x.rolling(5, min_periods=1).mean())

    print("\nShape posle dodavanja pokretnih proseka:")
    print("Train:", train.shape)
    print("Test:", test.shape)

    # PODELA NA TRAIN I VALIDACIONI SKUP
    # Delimo po motorima (po unit ID-u), ne po redovima, da izbegnemo curenje podataka.
    # Susedni ciklusi istog motora su skoro identicni, pa bi nasumicna podela redova
    # dala lazno dobre validacione metrike.
    # Koristimo poslednjih 20 motora (od 100) za validaciju -> 80/20 podela po motorima.
    svi_motori = train['unit'].unique()
    val_motori = svi_motori[-20:]
    train_motori = svi_motori[:-20]

    train_deo = train[train['unit'].isin(train_motori)]
    val_deo = train[train['unit'].isin(val_motori)]

    X_train = train_deo.drop(columns=['unit', 'ciklus', 'RUL'])
    y_train = train_deo['RUL']
    X_val = val_deo.drop(columns=['unit', 'ciklus', 'RUL'])
    y_val = val_deo['RUL']

    # Za test skup uzimamo poslednji poznati red svakog motora - jer zelimo da predvidimo
    # RUL na osnovu poslednjeg poznatog stanja
    X_test = test.groupby('unit').last().drop(columns=['ciklus'])
    y_test = rul['RUL']

    print("\nPodela po motorima (bez curenja podataka):")
    print(f"Train motori: {len(train_motori)}, Val motori: {len(val_motori)}")
    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_val shape:", y_val.shape)
    print("y_test shape:", y_test.shape)

    # CUVANJE OBRADJENIH PODATAKA
    # Cuvamo obradjene podatke u csv fajlove da ne moramo svaki put da cekamo obradu
    X_train.to_csv('data/X_train.csv', index=False)
    X_val.to_csv('data/X_val.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_val.to_csv('data/y_val.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)

    # Cuvamo i scaler da ga mozemo koristiti kasnije na novim podacima
    joblib.dump(scaler, 'modeli/scaler.pkl')

    print("\nPodaci sacuvani u CSV fajlove!")
    print("Scaler sacuvan kao modeli/scaler.pkl")

    return X_train, X_val, X_test, y_train, y_val, y_test


def ucitaj_test_za_vizualizacije():
    kolone = ['unit', 'ciklus', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
    test = pd.read_csv('data/test_FD001.txt', sep=r'\s+', header=None, names=kolone)
    return test


def ucitaj_za_vizualizacije():
    # Ucitavamo originalne podatke samo za vizualizacije
    # bez normalizacije i feature engineeringa
    kolone = ['unit', 'ciklus', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
    train = pd.read_csv('data/train_FD001.txt', sep=r'\s+', header=None, names=kolone)
    max_ciklus = train.groupby('unit')['ciklus'].max().reset_index()
    max_ciklus.columns = ['unit', 'max_ciklus']
    train = train.merge(max_ciklus, on='unit')
    train['RUL'] = train['max_ciklus'] - train['ciklus']
    train.drop(columns=['max_ciklus'], inplace=True)
    train['RUL'] = train['RUL'].clip(upper=125)
    return train