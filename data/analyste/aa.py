import requests
import pdfplumber
import re

url_pdf = "https://funds.amundi.com/dl/doc/monthly-factsheet/FR0011063353/FRA/FRA/RETAIL/CRCA/20251130"

# 1) Télécharger le PDF
resp = requests.get(url_pdf)
resp.raise_for_status()

with open("reporting.pdf", "wb") as f:
    f.write(resp.content)

# 2) Extraire le texte
full_text = ""
with pdfplumber.open("reporting.pdf") as pdf:
    for page in pdf.pages:
        full_text += page.extract_text() + "\n"

# 3) Récupérer la ligne "Performances glissantes" (Portefeuille …)
m_gliss = re.search(
    r"Performances glissantes.*?Portefeuille ([^\n]+)",
    full_text,
    flags=re.DOTALL
)
perf_glissantes = None
if m_gliss:
    valeurs_str = m_gliss.group(1)
    # Nettoyage : séparateur espace, virgule -> point, % enlevé
    valeurs = [
        float(v.replace('%', '').replace(',', '.'))
        for v in valeurs_str.split()
        if v.strip()  # exclure les vides
    ]
    # Amundi donne (d’après le PDF) : 1 mois, 3 mois, 1 an, 3 ans, 5 ans, 10 ans, depuis 2015, depuis création
    horizons = [
        "1m", "3m", "1y", "3y", "5y", "10y",
        "since_2015", "since_inception"
    ]
    perf_glissantes = dict(zip(horizons, valeurs))

# 4) Récupérer la ligne "Performances calendaires" (Portefeuille …)
m_cal = re.search(
    r"Performances calendaires.*?Portefeuille ([^\n]+)",
    full_text,
    flags=re.DOTALL
)
perf_calendaires = None
if m_cal:
    valeurs_str = m_cal.group(1)
    valeurs = [
        float(v.replace('%', '').replace(',', '.'))
        for v in valeurs_str.split()
        if v.strip()
    ]
    # D’après le PDF : 2024 2023 2022 2021 2020 2019 2018 2017 2016 2015
    annees = [2024, 2023, 2022, 2021, 2020,
              2019, 2018, 2017, 2016, 2015]
    perf_calendaires = dict(zip(annees, valeurs))

print("Performances glissantes :", perf_glissantes)
print("Performances calendaires :", perf_calendaires)

