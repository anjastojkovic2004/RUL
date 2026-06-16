import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib


def pripremi_sve():
    # =========================================================
    # UCITAVANJE PODATAKA
    # =========================================================
    # NASA C-MAPSS dataset ne sadrzi zaglavlja, pa ih rucno definisemo.
    # Struktura svake vrste: unit, ciklus, 3 operativna podesavanja, 21 senzor.
    # unit   -> jedinstveni ID motora (1-100)
    # ciklus -> redni broj leta (vremenska osa degradacije)
    # op1-3  -> operativna podesavanja: visina leta, brzina maha, ugao nagiba
    # s1-s21 -> merenja sa 21 senzora raspoređenih po motoru
    kolone = ['unit', 'ciklus', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]

    train = pd.read_csv('data/train_FD001.txt', sep=r'\s+', header=None, names=kolone)
    test = pd.read_csv('data/test_FD001.txt', sep=r'\s+', header=None, names=kolone)
    # RUL_FD001.txt sadrzi tacno 100 vrednosti - po jedna za svaki testni motor
    rul = pd.read_csv('data/RUL_FD001.txt', header=None, names=['RUL'])

    # train -> 20631 redova i 26 kolona
    # test  -> 13096 redova i 26 kolona
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    print("\nPrvih 5 redova train podataka:")
    print(train.head())
    print("\nPrvih 5 redova test podataka:")
    print(test.head())

    # =========================================================
    # PROVERA NEDOSTAJUCIH VREDNOSTI
    # =========================================================
    # NASA C-MAPSS dataset je simuliran i nema NaN vrednosti,
    # ali proveru uvek radimo kao deo dobre prakse u ML pipeline-u.
    # Ako bi NaN vrednosti postojale, morali bismo ih popuniti
    # (npr. linearnom interpolacijom) ili ukloniti te redove.
    nan_train = train.isnull().sum().sum()
    nan_test = test.isnull().sum().sum()

    print(f"\nBroj NaN vrednosti u train skupu: {nan_train}")
    print(f"Broj NaN vrednosti u test skupu: {nan_test}")

    if nan_train == 0 and nan_test == 0:
        print("Nema nedostajucih vrednosti - podaci su kompletni!")
    else:
        print("Postoje nedostajuce vrednosti - potrebno ciscenje!")

    # =========================================================
    # IZRACUNAVANJE RUL ZA TRENING SKUP
    # =========================================================
    # Trening skup prati svaki motor od pocetka rada do trenutka kvara (run-to-failure).
    # Poslednji ciklus svakog motora = trenutak kvara -> RUL = 0.
    # Formula: RUL = maksimalni_ciklus_motora - trenutni_ciklus
    # Primer: motor koji je radio 200 ciklusa, na ciklusu 150 ima RUL = 50.
    max_ciklus = train.groupby('unit')['ciklus'].max().reset_index()
    max_ciklus.columns = ['unit', 'max_ciklus']
    train = train.merge(max_ciklus, on='unit')
    train['RUL'] = train['max_ciklus'] - train['ciklus']
    train.drop(columns=['max_ciklus'], inplace=True)

    print("\nTrain sa RUL kolonom:")
    print(train[['unit', 'ciklus', 'RUL']].head(10))
    print("\nMaksimalni RUL:", train['RUL'].max())
    print("Minimalni RUL:", train['RUL'].min())

    # =========================================================
    # UKLANJANJE KONSTANTNIH SENZORA
    # =========================================================
    # Senzori cija standardna devijacija iznosi 0 ne menjaju vrednost ni kroz
    # jedan ciklus ni kroz jedan motor - nemaju informativnu vrednost za predikciju.
    # FD001 podskup ima jedan operativni rezim, pa su op3, s1, s10, s18, s19
    # uvek isti -> automatski ih detektujemo i uklanjamo iz oba skupa.
    # Vazno: uklanjamo ih iz test skupa jer trening i test moraju imati iste kolone.
    konstantni = [kol for kol in train.columns if train[kol].std() == 0]
    print("\nKonstantni senzori (uklanjamo ih):", konstantni)
    train.drop(columns=konstantni, inplace=True)
    test.drop(columns=konstantni, inplace=True)

    print("Train shape posle uklanjanja:", train.shape)
    print("Test shape posle uklanjanja:", test.shape)

    # =========================================================
    # NORMALIZACIJA
    # =========================================================
    # Senzori imaju razlicite opsege vrednosti (npr. temperatura 300-700,
    # pritisak 5-50). Bez normalizacije, senzori sa vecim opsegom bi dominirali
    # u modelu bez obzira na stvarnu vaznost.
    # MinMaxScaler prevodi sve vrednosti u opseg [0, 1].
    # KLJUCNO: scaler se trenira (fit) ISKLJUCIVO na train podacima.
    # Na test skupu se samo primenjuju nauceni parametri (transform) -
    # jer u realnosti ne bismo imali pristup test podacima unapred.
    senzori = [kol for kol in train.columns if kol not in ['unit', 'ciklus', 'RUL']]
    scaler = MinMaxScaler()
    train[senzori] = scaler.fit_transform(train[senzori])
    test[senzori] = scaler.transform(test[senzori])

    print("\nPodaci posle normalizacije (min/max trebaju biti 0/1):")
    print(train[senzori].describe().loc[['min', 'max']].round(2))

    # =========================================================
    # OGRANICAVANJE RUL NA 125 (Piecewise Linear Degradation)
    # =========================================================
    # Motori koji imaju RUL > 125 su daleko od kvara i senzori jos ne pokazuju
    # znake degradacije. Zbog toga je veoma tesko modelu da na osnovu senzora
    # razlikuje motor sa RUL=300 od motora sa RUL=200 - oba izgledaju "zdravo".
    # Standardni pristup u literaturi za FD001 dataset je postavljanje gornje
    # granice na 125 ciklusa: sve vrednosti vece od 125 se tretiraju kao 125.
    # Ovo poboljsava performanse modela jer fokusira ucenje na kriticnu fazu degradacije.
    RUL_MAX = 125
    train['RUL'] = train['RUL'].clip(upper=RUL_MAX)

    print("\nRUL posle ogranicavanja:")
    print("Maksimalni RUL:", train['RUL'].max())
    print("Minimalni RUL:", train['RUL'].min())

    # =========================================================
    # FEATURE ENGINEERING - POKRETNI PROSECI (Rolling Average)
    # =========================================================
    # Senzorska merenja su zasumljena - vrednost u jednom ciklusu moze biti
    # anomalija, a ne pravi signal degradacije. Pokretni prosek poslednjih
    # 5 ciklusa (window=5) gladi sum i istice pravi trend promene senzora.
    # Primer: ako s11 u ciklusima 10,11,12,13,14 ima vrednosti [0.3,0.35,0.28,0.32,0.36],
    #         pokretni prosek na ciklusu 14 je 0.322 - stabilniji signal.
    # min_periods=1 resava problem prvih ciklusa koji nemaju prethodnih 5 vrednosti.
    # Rezultat: broj atributa se duplira - za svaki senzor dodajemo _roll5 varijantu.
    senzori_bez_id = [kol for kol in train.columns if kol not in ['unit', 'ciklus', 'RUL']]
    for kol in senzori_bez_id:
        train[f'{kol}_roll5'] = train.groupby('unit')[kol].transform(lambda x: x.rolling(5, min_periods=1).mean())
        test[f'{kol}_roll5'] = test.groupby('unit')[kol].transform(lambda x: x.rolling(5, min_periods=1).mean())

    print("\nShape posle dodavanja pokretnih proseka:")
    print("Train:", train.shape)
    print("Test:", test.shape)

    # =========================================================
    # PODELA NA TRAIN I VALIDACIONI SKUP
    # =========================================================
    # Delimo PO MOTORIMA, a ne nasumicno po redovima.
    # Razlog: susedni ciklusi istog motora su skoro identicni po vrednostima senzora
    # i RUL-u (razlikuju se za 1). Nasumicna podela redova bi stavila ciklus 50 motora X
    # u train, a ciklus 51 istog motora u validaciju - model bi "znao odgovor" unapred.
    # Deljenjem po motorima osiguravamo da validacioni skup sadrzi motore koje
    # model nikad nije video tokom treninga - realnije merenje generalizacije.
    # Koristimo poslednjih 20 motora (81-100) za validaciju -> podela 80/20.
    svi_motori = train['unit'].unique()
    val_motori = svi_motori[-20:]
    train_motori = svi_motori[:-20]

    train_deo = train[train['unit'].isin(train_motori)]
    val_deo = train[train['unit'].isin(val_motori)]

    X_train = train_deo.drop(columns=['unit', 'ciklus', 'RUL'])
    y_train = train_deo['RUL']
    X_val = val_deo.drop(columns=['unit', 'ciklus', 'RUL'])
    y_val = val_deo['RUL']

    # Za test skup uzimamo POSLEDNJI poznati red svakog motora.
    # Test serije su namerno prekinute pre kvara - zadatak modela je da predvidi
    # koliko jos ciklusa motor moze da radi od tog poslednjeg poznatog stanja.
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

    # =========================================================
    # CUVANJE OBRADJENIH PODATAKA
    # =========================================================
    # Cuvamo obradjene skupove u CSV fajlove kako bi Streamlit aplikacija
    # mogla da ih ucita bez ponovnog pokretanja celog pipeline-a.
    # Scaler cuvamo kao .pkl fajl da bismo mogli da normalizujemo nove podatke
    # koristeci iste parametre koje smo naucili na trening skupu.
    X_train.to_csv('data/X_train.csv', index=False)
    X_val.to_csv('data/X_val.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_val.to_csv('data/y_val.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)

    joblib.dump(scaler, 'modeli/scaler.pkl')

    print("\nPodaci sacuvani u CSV fajlove!")
    print("Scaler sacuvan kao modeli/scaler.pkl")

    return X_train, X_val, X_test, y_train, y_val, y_test


def ucitaj_test_za_vizualizacije():
    # Ucitava sirove test podatke (bez obrade) iskljucivo za vizualizacije.
    # Potrebni su originalni ciklusi i redosled merenja za crtanje RUL krive.
    kolone = ['unit', 'ciklus', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
    test = pd.read_csv('data/test_FD001.txt', sep=r'\s+', header=None, names=kolone)
    return test


def ucitaj_za_vizualizacije():
    # Ucitava originalne train podatke sa izracunatim RUL-om, ali bez normalizacije
    # i feature engineeringa - jer za vizualizacije zelimo originalne vrednosti senzora
    # kako bi grafici bili fizicki interpretabilni (npr. prava temperatura, ne 0-1).
    kolone = ['unit', 'ciklus', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
    train = pd.read_csv('data/train_FD001.txt', sep=r'\s+', header=None, names=kolone)
    max_ciklus = train.groupby('unit')['ciklus'].max().reset_index()
    max_ciklus.columns = ['unit', 'max_ciklus']
    train = train.merge(max_ciklus, on='unit')
    train['RUL'] = train['max_ciklus'] - train['ciklus']
    train.drop(columns=['max_ciklus'], inplace=True)
    train['RUL'] = train['RUL'].clip(upper=125)
    return train
