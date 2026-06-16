# Predvidjanje preostalog zivotnog veka turboventilatorskih mlaznih motora (RUL)

Projekat razvoja regresionog modela masinskog ucenja za predikciju preostalog korisnog veka trajanja (RUL — Remaining Useful Life) turboventilatorskih mlaznih motora na osnovu podataka sa senzora, koristeci NASA C-MAPSS FD001 dataset.

**Autor:** Anja Stojkovic | RA 150/2023 | SAUSAU

---

## Sadrzaj

- [O projektu](#o-projektu)
- [Dataset](#dataset)
- [Instalacija](#instalacija)
- [Pokretanje](#pokretanje)
- [Struktura projekta](#struktura-projekta)
- [Metodologija](#metodologija)
- [Evaluacija](#evaluacija)

---

## O projektu

Cilj projekta je predikcija broja preostalih letova koje motor moze bezbedno da izvrsi pre kvara. Uspesna predikcija omogucava pravovremeno odrzavanje, smanjuje rizik od katastrofalnog kvara i optimizuje troskove servisiranja u avioindustriji.

Implementirani i uporedjeni modeli:
- **XGBoost** — glavni model
- **Random Forest** — model za poredjenje
- **Baseline** (predikcija proseka) — referentna tacka

---

## Dataset

NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation), podskup **FD001**:

| Fajl | Opis |
|------|------|
| `data/train_FD001.txt` | 100 motora pracenih do kvara — 20 631 redova |
| `data/test_FD001.txt` | 100 motora sa prekinutim vremenskim serijama — 13 096 redova |
| `data/RUL_FD001.txt` | Stvarne RUL vrednosti za finalnu evaluaciju |

Uslovi simulacije: jedan operativni rezim (nivo mora), jedan mod kvara (degradacija kompresora visokog pritiska).

---

## Instalacija

**Preduslovi:** Python 3.8+

```bash
pip install -r requirements.txt
```

Ili rucno:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib streamlit
```

---

## Pokretanje

### ML pipeline (priprema podataka, trening, evaluacija, vizualizacije)

```bash
python main.py
```

### Streamlit aplikacija

```bash
streamlit run app.py
```

---

## Struktura projekta

```
projekat_RUL/
├── data/
│   ├── train_FD001.txt       # Trening podaci
│   ├── test_FD001.txt        # Test podaci
│   └── RUL_FD001.txt         # Stvarne RUL vrednosti
├── grafici/                  # Generisane vizualizacije (.png)
├── modeli/                   # Sacuvani modeli i scaler (.pkl)
├── data.py                   # Ucitavanje i priprema podataka
├── model.py                  # Definicija i trening modela
├── vizualizacija.py          # Generisanje grafika
├── main.py                   # Glavni pipeline
├── app.py                    # Streamlit aplikacija
└── requirements.txt
```

---

## Metodologija

### Priprema podataka
1. Izracunavanje RUL po formuli: `RUL = max_ciklus - trenutni_ciklus`
2. Uklanjanje konstantnih senzora bez informativne vrednosti (op3, s1, s10, s18, s19)
3. Normalizacija vrednosti senzora — MinMaxScaler fitovan iskljucivo na trening skupu
4. Ogranicavanje RUL na 125 — sprecava dominaciju ranih ciklusa u ucenju
5. Feature engineering — pokretni proseci poslednjih 5 ciklusa (roll5) za svaki senzor
6. Podela po motorima: 80 motora za trening, 20 za validaciju (bez curenja podataka)

### Modeli
- XGBoost i Random Forest sa 200 stabala
- Podesavanje hiperparametara putem RandomizedSearchCV (20 iteracija, cv=3)
- Odabir top 10 najznacajnijih atributa na osnovu XGBoost feature importance

---

## Evaluacija

| Metrika | Opis |
|---------|------|
| **RMSE** | Root Mean Squared Error — standardno odstupanje predikcije od stvarnog RUL-a |
| **NASA Score** | Asimetricna funkcija gubitka — straze kažnjava kasna predvidjanja (motor koji "kasno otkaže") |
