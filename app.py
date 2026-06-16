import streamlit as st
import pandas as pd
import numpy as np
import joblib

model_xgb = joblib.load('modeli/xgboost_model.pkl')
model_kolone = model_xgb.feature_names_in_.tolist()

st.title("Predvidjanje preostalog zivotnog veka motora (RUL)")
st.write("Unesite vrednosti senzora da biste predvideli koliko jos ciklusa motor moze da radi.")

st.sidebar.header("Vrednosti senzora")

op1 = st.sidebar.slider("op1 — Visina leta (operativno podesavanje)", -0.01, 0.01, 0.0)
op2 = st.sidebar.slider("op2 — Mach broj (operativno podesavanje)", -0.01, 0.01, 0.0)
s2 = st.sidebar.slider("s2 — T24: Temperatura na izlazu iz LPC", 0.0, 1.0, 0.5)
s3 = st.sidebar.slider("s3 — T30: Temperatura na izlazu iz HPC", 0.0, 1.0, 0.5)
s4 = st.sidebar.slider("s4 — T50: Temperatura na izlazu iz LPT ⚠️", 0.0, 1.0, 0.5)
s5 = st.sidebar.slider("s5 — P2: Pritisak na ulazu u fan", 0.0, 1.0, 0.5)
s6 = st.sidebar.slider("s6 — P15: Pritisak u bypass-kanalu", 0.0, 1.0, 0.5)
s7 = st.sidebar.slider("s7 — P30: Pritisak na izlazu iz HPC ⚠️", 0.0, 1.0, 0.5)
s8 = st.sidebar.slider("s8 — Nf: Fizicka brzina vrtnje fana", 0.0, 1.0, 0.5)
s9 = st.sidebar.slider("s9 — Nc: Fizicka brzina vrtnje jezgra ⚠️", 0.0, 1.0, 0.5)
s11 = st.sidebar.slider("s11 — Ps30: Staticki pritisak na izlazu iz HPC ⚠️", 0.0, 1.0, 0.5)
s12 = st.sidebar.slider("s12 — phi: Odnos protoka goriva i pritiska ⚠️", 0.0, 1.0, 0.5)
s13 = st.sidebar.slider("s13 — NRf: Korigovana brzina fana", 0.0, 1.0, 0.5)
s14 = st.sidebar.slider("s14 — NRc: Korigovana brzina jezgra", 0.0, 1.0, 0.5)
s15 = st.sidebar.slider("s15 — BPR: Bypass odnos ⚠️", 0.0, 1.0, 0.5)
s16 = st.sidebar.slider("s16 — farB: Odnos gorivo-vazduh", 0.0, 1.0, 0.5)
s17 = st.sidebar.slider("s17 — htBleed: Bleed vazduh za hladjenje", 0.0, 1.0, 0.5)
s20 = st.sidebar.slider("s20 — W31: Protok vazduha za hladjenje HPT", 0.0, 1.0, 0.5)
s21 = st.sidebar.slider("s21 — W32: Protok vazduha za hladjenje LPT", 0.0, 1.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.markdown("**⚠️ Kljucni senzori (najveca vaznost u modelu):**")
st.sidebar.markdown("- **s4 (T50)** raste → motor se kvari")
st.sidebar.markdown("- **s11 (Ps30)** raste → motor se kvari")
st.sidebar.markdown("- **s9 (Nc)** raste → motor se kvari")
st.sidebar.markdown("- **s12 (phi)** opada → motor se kvari")
st.sidebar.markdown("- **s7 (P30)** opada → motor se kvari")
st.sidebar.markdown("- **s15 (BPR)** opada → motor se kvari")

mapa = {
    'op1': op1, 'op2': op2, 's2': s2, 's3': s3, 's4': s4,
    's5': s5, 's6': s6, 's7': s7, 's8': s8, 's9': s9,
    's11': s11, 's12': s12, 's13': s13, 's14': s14,
    's15': s15, 's16': s16, 's17': s17, 's20': s20, 's21': s21
}

podaci = pd.DataFrame(0.0, index=[0], columns=model_kolone)

for kol, val in mapa.items():
    if kol in podaci.columns:
        podaci[kol] = val
    if f'{kol}_roll5' in podaci.columns:
        podaci[f'{kol}_roll5'] = val

rul = model_xgb.predict(podaci)[0]
rul = max(0, min(125, rul))

st.header("Rezultat predikcije")

if rul > 80:
    st.success(f"Motor je ZDRAV — preostalo jos oko {rul:.0f} ciklusa")
elif rul > 30:
    st.warning(f"Motor pokazuje ZNAKE DEGRADACIJE — preostalo jos oko {rul:.0f} ciklusa")
else:
    st.error(f"Motor je BLIZU KVARA — preostalo samo oko {rul:.0f} ciklusa")

st.metric("Predvidjeni RUL", f"{rul:.0f} ciklusa")
st.progress(int(rul / 125 * 100))

st.markdown("---")
st.subheader("Objasnjenje senzora")
tabela = pd.DataFrame({
    'Senzor': ['op1', 'op2', 's2', 's3', 's4', 's5', 's6', 's7', 's8',
               's9', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's20', 's21'],
    'Naziv': ['Altitude', 'Mach', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf',
              'Nc', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'W31', 'W32'],
    'Opis': [
        'Visina leta', 'Mach broj leta',
        'Temperatura na izlazu iz LPC', 'Temperatura na izlazu iz HPC',
        'Temperatura na izlazu iz LPT', 'Pritisak na ulazu u fan',
        'Pritisak u bypass-kanalu', 'Pritisak na izlazu iz HPC',
        'Brzina vrtnje fana', 'Brzina vrtnje jezgra',
        'Staticki pritisak na izlazu iz HPC', 'Odnos protoka goriva i pritiska',
        'Korigovana brzina fana', 'Korigovana brzina jezgra',
        'Bypass odnos', 'Odnos gorivo-vazduh',
        'Bleed vazduh za hladjenje HPT', 'Protok vazduha za hladjenje HPT',
        'Protok vazduha za hladjenje LPT'
    ],
    'Uticaj na kvar': [
        '-', '-', '-', '-',
        'Raste ⚠️', '-', '-',
        'Opada ⚠️', '-',
        'Raste ⚠️', 'Raste ⚠️', 'Opada ⚠️',
        '-', '-', 'Opada ⚠️', '-', '-', '-', '-'
    ]
})
st.table(tabela)

st.markdown("---")
st.subheader("Uputstvo za koriscenje")
st.markdown("""
**Kako koristiti aplikaciju:**

1. **Podesiti klizace** sa leve strane prema vrednostima senzora vaseg motora
2. **Pratiti rezultat** koji se automatski azurira pri svakoj promeni
3. **Tumaciti rezultat:**
   - 🟢 **Zdrav motor (RUL > 80)** — motor radi normalno
   - 🟡 **Znaci degradacije (30 < RUL < 80)** — planirati odrzavanje
   - 🔴 **Blizu kvara (RUL < 30)** — hitno odrzavanje neophodno!

**Napomena o vrednostima klizaca:**
Klizaci su u normalizovanom opsegu 0-1, gde 0 = minimalna izmerena vrednost
senzora u datasetu, 1 = maksimalna, a 0.5 = srednja vrednost.

**Fizicka osnova:** Senzori oznaceni sa ⚠️ direktno odrazavaju termodinamicku
efikasnost kompresora i turbina. Habanje lopatica usled visokih temperatura
i pritisaka uzrokuje promene ovih vrednosti tokom zivotnog veka motora.

**Napomena:** Ova aplikacija koristi XGBoost model istreniran na NASA C-MAPSS
FD001 datasetu. Predikcija sluzi kao podrska odlucivanju, ne kao zamena za
strucni tehnicki pregled motora.
""")

st.markdown("---")
st.caption("Razvijeno u okviru projekta SAUSAU | NASA C-MAPSS FD001 Dataset | FTN Novi Sad 2026")