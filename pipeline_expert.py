import argparse
import csv
import json
import logging
import os
import re
import subprocess
import threading
import time
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import curses
import pdfplumber
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

DATA_DIR = Path("data/dic_pdfs")
OUTPUT_CSV = Path("data/dic_summary_expert.csv")
GLOBAL_CSV = Path("data/dic_summary_expert_all.csv")
DOWNLOAD_LOG = Path("data/download_log.csv")
DEFAULT_URL = "https://priips.predica.com/credit-agricole/consultation-support"
MISSING = "MISSING"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
ASSISTANT_MODEL = os.getenv("OPENAI_ASSISTANT_MODEL", "gpt-4o-mini")
MAX_OPENAI_CHARS = int(os.getenv("OPENAI_MAX_CHARS", "50000"))
ALLOWED_ANALYST_MANAGERS = ["AMUNDI"]
ANALYST_MANAGER_URLS = {
    "AMUNDI": "https://funds.amundi.com/dl/doc/monthly-factsheet/{isin}/FRA/FRA/RETAIL/CRCA",
}
ANALYST_REPORT_DIR = Path("data/analyste")
ASSISTANT_ID_FILE = ANALYST_REPORT_DIR / "assistant_id.txt"



def _is_allowed_manager(name: Optional[str]) -> bool:
    if not name:
        return False
    lower = name.lower()
    return any(allowed.lower() in lower for allowed in ALLOWED_ANALYST_MANAGERS)


def _find_manager_by_isin(isin: str) -> Optional[str]:
    if not GLOBAL_CSV.exists():
        return None
    try:
        with open(GLOBAL_CSV, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row.get("isin", "")).upper() == isin.upper():
                    return row.get("management_company")
    except Exception as exc:
        logging.error("Recherche société de gestion par ISIN échouée: %s", exc)
    return None


def _analyst_report(isin: str, management_company: Optional[str] = None, debug: bool = False) -> str:
    if not isin or isin == MISSING:
        return "ISIN manquant pour générer un rapport."
    if not management_company:
        management_company = _find_manager_by_isin(isin)
    _ensure_reporting_pdf(isin, management_company)
    try:
        return _run_assistant_on_reporting(isin, management_company, debug=debug)
    except Exception as exc:
        logging.error("Échec du rapport analyste pour %s: %s", isin, exc)
        return f"Erreur lors de la génération du rapport pour {isin}: {exc}"


def _ensure_reporting_pdf(isin: str, management_company: str) -> None:
    ANALYST_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    dest = ANALYST_REPORT_DIR / f"reporting-{isin}.pdf"
    if dest.exists():
        return
    url_template = None
    for key, url in ANALYST_MANAGER_URLS.items():
        if key.lower() in (management_company or "").lower():
            url_template = url
            break
    if not url_template:
        logging.info("No reporting URL configured for manager %s", management_company)
        return
    url = url_template.format(isin=isin)
    logging.debug(
        "Reporting absent. Veuillez télécharger manuellement le rapport pour %s depuis %s et le sauvegarder sous %s",
        isin,
        url,
        dest,
    )


def _download_with_spinner(url: str, dest: Path, timeout: int = 30) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["curl", "-sSL", "--max-time", str(timeout), "-o", str(dest), url],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _download_url_to(dest: Path, url: str, timeout: int = 30) -> bool:
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["curl", "-sSL", "--max-time", str(timeout), "-o", str(dest), url],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            if dest.stat().st_size > 1_000_000:
                logging.error("Fichier trop volumineux (>1Mo) pour %s", dest)
                dest.unlink(missing_ok=True)
                return False
        except Exception:
            pass
        return True
    except Exception as exc:
        logging.error("Échec du téléchargement manuel %s -> %s: %s", url, dest, exc)
        return False


def _run_assistant_on_reporting(isin: str, management_company: str, debug: bool = False) -> str:
    pdf_path = ANALYST_REPORT_DIR / f"reporting-{isin}.pdf"
    if not pdf_path.exists():
        logging.debug("Reporting introuvable pour %s", isin)
        return (
            f"Reporting introuvable. Merci de télécharger manuellement le dernier rapport pour l'ISIN {isin} "
            f"et de le sauvegarder sous {pdf_path}"
        )
    cached_report = ANALYST_REPORT_DIR / f"rapport-{isin}.txt"
    if cached_report.exists():
        try:
            text = cached_report.read_text()
            return f"[rapport existant]\n\n{text}"
        except Exception:
            pass
    client = OpenAI()
    if debug:
        logging.debug("Upload du reporting %s vers OpenAI", pdf_path)
    with open(pdf_path, "rb") as f:
        upload = client.files.create(file=f, purpose="assistants")
    instructions = (
         "Tu es un assistant financier. "
        "À partir de reportings mensuels au format PDF, tu dois extraire : "
        "1) les performances glissantes du portefeuille, affiché verticalement, pas en mode tableau, "
        "2) les performances calendaires par année, affiché verticalement, pas en mode tableau,"
        "3) un résumé clair pour un client retail. tu dois emettre des avis objectifs basés sur les données. et décrire si cela ressemble à un bon fonds ou pas. "
        " you must format clearly the output to be visible on a unix terminal with proper tabulation"
    )
    assistant_id = None
    if ASSISTANT_ID_FILE.exists():
        try:
            assistant_id = ASSISTANT_ID_FILE.read_text().strip()
        except Exception:
            assistant_id = None
    if not assistant_id:
        assistant = client.beta.assistants.create(
            name="Analyse de reporting",
            instructions=instructions,
            model=ASSISTANT_MODEL,
            tools=[{"type": "file_search"}],
        )
        ASSISTANT_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
        ASSISTANT_ID_FILE.write_text(assistant.id)
        assistant_id = assistant.id
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": f"Analyse le reporting PDF pour l'ISIN {isin}. Donne un résumé concis.",
                "attachments": [{"file_id": upload.id, "tools": [{"type": "file_search"}]}],
            }
        ]
    )
    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant_id)
    while run.status in ("queued", "in_progress", "requires_action"):
        time.sleep(0.5)
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
    if run.status != "completed":
        return f"Analyse échouée (statut {run.status})"
    messages = client.beta.threads.messages.list(thread_id=thread.id, order="desc", limit=1)
    analysis = ""
    for msg in messages.data:
        for content in msg.content:
            if content.type == "text":
                analysis = content.text.value
                break
        if analysis:
            break
    if not analysis:
        return "Aucune réponse générée."
    dest = ANALYST_REPORT_DIR / f"rapport-{isin}.txt"
    with open(dest, "w") as f:
        f.write(analysis)
    try:
        client.files.delete(upload.id)
    except Exception as exc:
        logging.warning("Impossible de supprimer le fichier OpenAI %s: %s", upload.id, exc)
    return analysis



def fetch_pdf_links(base_url: str) -> List[str]:
    resp = requests.get(base_url, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    pdf_urls = set()
    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        if href and href.lower().endswith(".pdf"):
            pdf_urls.add(urljoin(base_url, href))
    return sorted(pdf_urls)


def _safe_filename(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name or "document.pdf"
    return name.replace("%20", "_")


def download_pdfs(urls: List[str], data_dir: Path = DATA_DIR) -> List[Dict[str, str]]:
    data_dir.mkdir(parents=True, exist_ok=True)
    log_rows = []
    for url in urls:
        filename = _safe_filename(url)
        dest = data_dir / filename
        downloaded = False
        try:
            if dest.exists():
                logging.debug("Skip existing %s", dest)
            else:
                with requests.get(url, stream=True, timeout=30) as resp:
                    resp.raise_for_status()
                    with open(dest, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                downloaded = True
                logging.debug("Downloaded %s", dest)
            size = dest.stat().st_size if dest.exists() else 0
        except Exception as exc:
            logging.error("Failed %s: %s", url, exc)
            size = 0
        log_rows.append(
            {
                "url": url,
                "file_name": filename if dest.exists() else MISSING,
                "size_bytes": size,
                "downloaded_at": datetime.utcnow().isoformat(),
                "status": "downloaded" if downloaded else ("exists" if dest.exists() else "error"),
            }
        )
    _append_log(log_rows)
    return log_rows


def _append_log(rows: List[Dict[str, str]]) -> None:
    DOWNLOAD_LOG.parent.mkdir(parents=True, exist_ok=True)
    exists = DOWNLOAD_LOG.exists()
    with open(DOWNLOAD_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "file_name", "size_bytes", "downloaded_at", "status"])
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def extract_text_from_pdf(pdf_path: Path) -> Optional[str]:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = [page.extract_text(layout=True) or "" for page in pdf.pages]
        return "\n".join(pages)
    except Exception as exc:
        logging.error("Cannot read %s: %s", pdf_path, exc)
        return None


def _find_isin(text: str) -> str:
    match = re.search(r"\b([A-Z]{2}[A-Z0-9]{10})\b", text)
    return match.group(1) if match else MISSING


def _has_isin_on_first_page(pdf_path: Path) -> bool:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                return False
            first_text = pdf.pages[0].extract_text(layout=True) or ""
            return _find_isin(first_text) != MISSING
    except Exception as exc:
        logging.error("Failed first-page ISIN check for %s: %s", pdf_path, exc)
        return False


def parse_pdf_fields_with_openai(pdf_path: Path, debug: bool = False) -> Dict[str, str]:
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return _empty_fields(pdf_path)
    if len(text) > MAX_OPENAI_CHARS:
        logging.info("Texte tronqué pour %s (len=%d > %d)", pdf_path.name, len(text), MAX_OPENAI_CHARS)
        text = text[:MAX_OPENAI_CHARS]

    client = OpenAI()
    prompt = textwrap.dedent(
        """
        Tu es une IA spécialisée dans l’analyse de Documents d’Informations Clés (DIC / KID PRIIPs) pour fonds et unités de compte.

On va te fournir le texte brut d’un DIC (sans mise en page PDF, issu d’un PDF).

Ton objectif est d’extraire les informations utiles à l’investisseur et de produire UN SEUL objet JSON valide, structuré en deux sections :

- "simple" : informations essentielles
- "expert" : informations avancées d’aide à la décision

Tu dois répondre UNIQUEMENT avec ce JSON, sans commentaire, sans texte autour, sans balises Markdown.

----------------------------------------------------
1. STRUCTURE GLOBALE ATTENDUE
----------------------------------------------------

Tu dois toujours renvoyer une structure JSON de ce type (la structure, pas les valeurs) :

{
  "simple": {
    "product_name": "...",
    "isin": "...",
    "sri": 4,
    "horizon": 5,
    "frais_courants_pct": 2.07,
    "frais_entree_pct": 2,
    "frais_sortie_pct": 0
  },
  "expert": {
    "asset_class": "...",
    "investment_region": "...",
    "management_style": "...",
    "objective_summary": "...",
    "benchmark": "...",
    "sfdr_classification": "...",
    "main_risks": [],
    "nav_frequency": null,
    "liquidity_constraints": null,
    "performance_fee_pct": 0,
    "ongoing_fees_breakdown": {
      "management_fees_pct": null,
      "transaction_costs_pct": null,
      "other_costs_pct": null
    },
    "currency": "...",
    "management_company": "...",
    "costs_vs_fees_note": "..."
  }
}

Contraintes :
- toutes les clés doivent exister, même si la valeur est null
- les nombres doivent être de vrais nombres JSON (sans guillemets)
- pas d’autre clé que celles listées ci-dessous.

----------------------------------------------------
2. SECTION "simple" — EXTRACTION DE BASE
----------------------------------------------------

Champs obligatoires :

- "product_name"
- "isin"
- "sri"
- "horizon"
- "frais_courants_pct"
- "frais_entree_pct"
- "frais_sortie_pct"

Règles générales :
- Si une information est introuvable ou vraiment ambiguë → null (sauf là où on demande explicitement 0).
- N’écris jamais "null" en chaîne : utilise le littéral JSON null.

Règles par champ :

1) product_name
Repère le nom exact dans la section "Produit" ou "Nom du produit".
Ne modifie pas le texte (pas de reformulation). Si non trouvé → null.

2) isin
Repère "ISIN" ou "Code ISIN".
Extrais le premier code ISIN à 12 caractères alphanumériques. Si absent → null.

3) sri
Repère une phrase du type :
- "classe de risque X sur 7"
- ou "X sur une échelle de 1 à 7"
Renvoie uniquement le chiffre X (entier). Si rien d’exploitable → null.

4) horizon
Repère la durée de placement recommandée :
- "Période de détention recommandée : … ans"
- ou "Horizon de placement recommandé : … ans"
Renvoie le nombre d’années (entier). Si introuvable → null.

----------------------------------------------------
2.3 Coûts vs frais (important)
----------------------------------------------------

Dans les DIC PRIIPs, le mot "coûts" est plus large que "frais".
Pour ce prompt, on adopte la convention suivante :

a) frais_courants_pct

"frais_courants_pct" doit représenter le total des coûts récurrents annuels du produit, au sens PRIIPs, c’est-à-dire :
- frais de gestion et autres frais administratifs / d’exploitation
- + coûts de transaction
- + autres coûts récurrents éventuels inclus dans le DIC

Règles d’extraction :
1. Si le DIC explicite une ligne du type "Incidence des coûts annuels" ou "Coûts annuels totaux" en %, tu peux l’utiliser comme approximation si elle représente bien l’ensemble des coûts récurrents du produit (et pas seulement la distribution).
2. Sinon, additionne explicitement :
   - "Frais de gestion et autres coûts administratifs ou d’exploitation"
   - + "Coûts de transaction"
   - + autres lignes récurrentes si elles existent.
3. Convertis en pourcentage numérique sans signe % (ex : "1,45 %" + "0,62 %" → 2.07).

Si aucune information fiable ne permet de calculer cela → null.

b) frais_entree_pct

Correspond au coût ponctuel d’entrée maximal facturé à l’investisseur.
- Si le DIC indique "Coûts d’entrée … jusqu’à X %" → renvoyer X.
- Si "Nous ne facturons pas de coûts d’entrée" ou "0 %" → 0.
- Si le texte ne permet pas de déduire un pourcentage clair → null.

c) frais_sortie_pct

Même logique que pour les coûts d’entrée, mais sur les "Coûts de sortie".
- "Nous ne facturons pas de coûts de sortie" ou "0 %" → 0.
- "X %" → renvoyer X.
- Sinon → null.

----------------------------------------------------
3. SECTION "expert" — INFORMATIONS AVANCÉES
----------------------------------------------------

Champs obligatoires :

- "asset_class"
- "investment_region"
- "management_style"
- "objective_summary"
- "benchmark"
- "sfdr_classification"
- "main_risks"
- "nav_frequency"
- "liquidity_constraints"
- "performance_fee_pct"
- "ongoing_fees_breakdown"
- "currency"
- "management_company"
- "costs_vs_fees_note"

Règles par champ :

asset_class
Classe d’actifs dominante, par ex. :
- "Actions", "Obligations", "Immobilier", "Diversifié", "Monétaire", "Inconnu".

investment_region
Zone d’investissement principale : ex. "Europe", "Monde", "États-Unis", "Émergents", "Monde (hors Zone Euro)", etc.

management_style
- "Active", "Indicielle", ou "Mixte".

objective_summary
Court résumé factuel (1–3 phrases) de l’objectif et de la stratégie du produit, basé sur le DIC. Pas d’invention.

benchmark
Nom de l’indice de référence mentionné (par ex. "MSCI World ex EMU Selection").
Si aucun indice → null.

sfdr_classification
En fonction du DIC :
- "SFDR Article 8"
- "SFDR Article 9"
- "Non classé / Non mentionné"

main_risks
Liste JSON (array) des principaux risques spécifiques cités, par ex. :
"Risque de marché", "Risque de perte en capital", "Risque de liquidité", "Risque lié aux dérivés", etc.

nav_frequency
Fréquence de calcul de la valeur liquidative :
- "Quotidienne", "Hebdomadaire", "Mensuelle", etc.
Si non clair → null.

liquidity_constraints
Texte décrivant les contraintes de liquidité / rachat :
- mécanisme de "gates",
- conditions de sortie spécifiques (SCPI, marché secondaire, délai long, etc.).
Si rien de particulier → null.

performance_fee_pct
Pourcentage de commission de surperformance si elle existe.
- S’il est indiqué qu’il n’y a pas de commission de performance → 0.
- S’il y a une commission de performance exprimée en % → renvoyer ce % (nombre).
- Si ce n’est pas clair → 0 par défaut.

ongoing_fees_breakdown
Objet JSON détaillant les composantes des coûts récurrents annuels :

{
  "management_fees_pct": 1.45,
  "transaction_costs_pct": 0.62,
  "other_costs_pct": null
}

- "management_fees_pct" : frais de gestion et autres frais administratifs / d’exploitation, en %.
- "transaction_costs_pct" : coûts de transaction, en %.
- "other_costs_pct" : autres coûts récurrents éventuels (en %), sinon null.

Les pourcentages doivent être cohérents avec ceux utilisés pour "frais_courants_pct".
Si un élément n’est pas isolable → null.

currency
Devise de référence du fonds (ex. "EUR", "USD"). Si non indiqué → null.

management_company
Nom de la société de gestion (ex. "Amundi Asset Management", "AXA Investment Managers").

costs_vs_fees_note
Brève explication textuelle en français, dans tes propres mots, clarifiant pour l’investisseur la différence entre :
- les coûts récurrents ("frais_courants_pct"),
- les frais ponctuels ("frais_entree_pct", "frais_sortie_pct"),
- et, le cas échéant, la commission de performance.

Elle doit rappeler que :
- les coûts récurrents sont prélevés chaque année et réduisent la performance,
- les frais d’entrée/sortie s’appliquent à l’achat/vente,
- certains coûts (comme les coûts de transaction) ne sont pas facturés directement mais réduisent le rendement.

----------------------------------------------------
4. CONTRAINTES DE FORMAT (JSON STRICT)
----------------------------------------------------

- Ta réponse doit être un unique JSON valide.
- Pas de texte avant, pas de texte après.
- Pas de commentaires, pas de Markdown, pas de backticks.
- Les champs numériques doivent être des nombres JSON (pas de chaînes).
- Utilise null pour les valeurs manquantes (et non "null").

Si une information n’est pas dans le texte, tu ne l’inventes pas : tu mets null ou 0 selon les règles ci-dessus.

        """
    )
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "user", "content": text},
            ],
            temperature=0,
        )
        content = resp.choices[0].message.content or ""
        if debug:
            logging.info("OpenAI raw response for %s: %s", pdf_path.name, content)
        base, expert = _parse_base_expert(content, pdf_path)
        if base is None and expert is None:
            return _empty_fields(pdf_path)
        return _merge_base_expert(base, expert, pdf_path)
    except Exception as exc:
        logging.error("OpenAI parsing failed for %s: %s", pdf_path, exc)
        return _empty_fields(pdf_path)


def _merge_base_expert(base: Dict[str, any], expert: Dict[str, any], pdf_path: Path) -> Dict[str, str]:
    def get_val(d, key):
        return d.get(key, None) if isinstance(d, dict) else None

    def num_or_missing(v):
        return v if isinstance(v, (int, float)) else MISSING if v is None else v

    ongoing = expert.get("ongoing_fees_breakdown", {}) if isinstance(expert, dict) else {}
    if isinstance(ongoing, list) and ongoing:
        ongoing = ongoing[0]
    return {
        "product_name": base.get("product_name", MISSING),
        "isin": base.get("isin", MISSING),
        "sri": str(base.get("sri", MISSING)),
        "horizon": base.get("horizon", MISSING),
        "frais_courants_pct": num_or_missing(base.get("frais_courants_pct")),
        "frais_entree_pct": num_or_missing(base.get("frais_entree_pct")),
        "frais_sortie_pct": num_or_missing(base.get("frais_sortie_pct")),
        "asset_class": get_val(expert, "asset_class") or MISSING,
        "investment_region": get_val(expert, "investment_region") or MISSING,
        "management_style": get_val(expert, "management_style") or MISSING,
        "objective_summary": get_val(expert, "objective_summary") or MISSING,
        "benchmark": get_val(expert, "benchmark") or MISSING,
        "sfdr_classification": get_val(expert, "sfdr_classification") or MISSING,
        "main_risks": "; ".join(expert.get("main_risks", [])) if isinstance(expert.get("main_risks", []), list) else MISSING,
        "nav_frequency": get_val(expert, "nav_frequency") or MISSING,
        "liquidity_constraints": get_val(expert, "liquidity_constraints") or MISSING,
        "performance_fee_pct": num_or_missing(get_val(expert, "performance_fee_pct")),
        "management_fees_pct": num_or_missing(ongoing.get("management_fees_pct")),
        "transaction_costs_pct": num_or_missing(ongoing.get("transaction_costs_pct")),
        "other_costs_pct": num_or_missing(ongoing.get("other_costs_pct")),
        "currency": get_val(expert, "currency") or MISSING,
        "management_company": get_val(expert, "management_company") or MISSING,
        "source_pdf": str(pdf_path),
    }


def _parse_base_expert(content: str, pdf_path: Path) -> (Optional[Dict[str, any]], Optional[Dict[str, any]]):
    """Parse OpenAI response that may contain two JSON objects or a dict with simple/base + expert."""
    cleaned = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()

    # 1) Direct JSON load
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            base = data.get("base") or data.get("simple") or data.get("simp")
            expert = data.get("expert")
            if base is not None or expert is not None:
                return base, expert
    except Exception:
        pass

    # 2) Try to extract two JSON objects from text (first is base/simple, second expert)
    objs = re.findall(r"\{(?:[^{}]|(?R))*\}", cleaned, flags=re.DOTALL)
    if len(objs) >= 2:
        try:
            base_obj = json.loads(objs[0])
            expert_obj = json.loads(objs[1])
            return base_obj, expert_obj
        except Exception:
            logging.error("Failed to parse dual JSON objects for %s; snippets: %s ... %s", pdf_path, objs[0][:120], objs[1][:120])
            return None, None

    logging.error("OpenAI parsing failed for %s; raw: %s", pdf_path, content[:200])
    return None, None


def _empty_fields(pdf_path: Path) -> Dict[str, str]:
    return {
        "product_name": MISSING,
        "isin": MISSING,
        "sri": MISSING,
        "horizon": MISSING,
        "frais_courants_pct": MISSING,
        "frais_entree_pct": MISSING,
        "frais_sortie_pct": MISSING,
        "asset_class": MISSING,
        "investment_region": MISSING,
        "management_style": MISSING,
        "objective_summary": MISSING,
        "benchmark": MISSING,
        "sfdr_classification": MISSING,
        "main_risks": MISSING,
        "nav_frequency": MISSING,
        "liquidity_constraints": MISSING,
        "performance_fee_pct": MISSING,
        "management_fees_pct": MISSING,
        "transaction_costs_pct": MISSING,
        "other_costs_pct": MISSING,
        "currency": MISSING,
        "management_company": MISSING,
        "source_pdf": str(pdf_path),
    }


def parse_pdf_fields(pdf_path: Path, debug: bool = False) -> Dict[str, str]:
    return parse_pdf_fields_with_openai(pdf_path, debug=debug)


def process_all_pdfs(
    limit: Optional[int] = None,
    debug: bool = False,
    pdf_paths: Optional[List[Path]] = None,
    data_dir: Path = DATA_DIR,
    output_csv: Path = OUTPUT_CSV,
) -> List[Dict[str, str]]:
    rows = []
    existing = set()
    if output_csv.exists():
        with open(output_csv, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing.add(Path(r.get("source_pdf", "")).name)
    if pdf_paths is None:
        pdf_paths = sorted(data_dir.glob("*.pdf"))
    parsed_count = 0
    for pdf_path in pdf_paths:
        if pdf_path.name in existing:
            logging.info("Skipping %s (already in CSV)", pdf_path.name)
            continue
        if not _has_isin_on_first_page(pdf_path):
            logging.info("Skipping %s (no ISIN on first page)", pdf_path)
            continue
        fields = parse_pdf_fields(pdf_path, debug=debug)
        rows.append(fields)
        parsed_count += 1
        if limit is not None and parsed_count >= limit:
            break
    return rows


FIELDNAMES = [
    "product_name",
    "isin",
    "sri",
    "horizon",
    "frais_courants_pct",
    "frais_entree_pct",
    "frais_sortie_pct",
    "asset_class",
    "investment_region",
    "management_style",
    "objective_summary",
    "benchmark",
    "sfdr_classification",
    "main_risks",
    "nav_frequency",
    "liquidity_constraints",
    "performance_fee_pct",
    "management_fees_pct",
    "transaction_costs_pct",
    "other_costs_pct",
    "currency",
    "management_company",
    "source_pdf",
    "fond",
]


def write_output(rows: List[Dict[str, str]], output_csv: Path = OUTPUT_CSV, fond_label: str = "default") -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = output_csv.exists()
    enriched = []
    for row in rows:
        row = dict(row)
        row.setdefault("fond", fond_label)
        enriched.append(row)
    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerows(enriched)
    _append_global(enriched)


def _append_global(rows: List[Dict[str, str]]) -> None:
    GLOBAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    exists = GLOBAL_CSV.exists()
    with open(GLOBAL_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def load_output_rows(output_csv: Path = OUTPUT_CSV) -> List[Dict[str, str]]:
    if not output_csv.exists():
        logging.warning("Output CSV %s not found", output_csv)
        return []
    with open(output_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            r = dict(r)
            if not r.get("fond"):
                try:
                    parent = Path(r.get("source_pdf", "")).parent.name
                    if parent and parent != "data":
                        r["fond"] = parent
                    else:
                        r["fond"] = "default"
                except Exception:
                    r["fond"] = "default"
            rows.append(r)
        return rows


def regenerate_global_csv() -> None:
    GLOBAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    sources = {OUTPUT_CSV}
    for csv_path in Path("data").glob("*/dic_summary_expert_*.csv"):
        if csv_path.resolve() == GLOBAL_CSV.resolve():
            continue
        sources.add(csv_path)
    rows: List[Dict[str, str]] = []
    for src in sources:
        if not src.exists():
            continue
        fond_label = src.parent.name if src.parent.name != "data" else "default"
        try:
            with open(src, newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    r = dict(r)
                    r.setdefault("fond", fond_label)
                    rows.append(r)
        except Exception as exc:
            logging.error("Failed to read %s: %s", src, exc)
    with open(GLOBAL_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    logging.info("Regenerated %s with %d rows", GLOBAL_CSV, len(rows))


def _risk_color_pair(risk: str) -> int:
    try:
        value = int(risk)
    except Exception:
        return 0
    if value <= 2:
        return 1
    if value <= 4:
        return 2
    if value <= 5:
        return 3
    return 4


def display_with_curses(rows: List[Dict[str, str]], debug: bool = False) -> None:
    def _inner(stdscr):
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_GREEN)
        curses.curs_set(0)

        headers = ["Fond", "Produit", "ISIN", "SRI", "Horizon", "Frais courants", "Entrée", "Sortie"]
        col_widths = [12, 25, 14, 5, 25, 14, 10, 10]
        selected = 0
        top_offset = 0
        rows_list = list(rows)

        def draw_row(y: int, cols: List[str], highlight_risk: Optional[str] = None, is_selected: bool = False):
            x = 1
            color = curses.color_pair(_risk_color_pair(highlight_risk) if highlight_risk else 0)
            if is_selected:
                color |= curses.A_REVERSE
            for idx, col in enumerate(cols):
                col_str = "" if col is None else str(col)
                cell = col_str[: col_widths[idx]].ljust(col_widths[idx])
                stdscr.addstr(y, x, cell, color)
                x += col_widths[idx] + 1

        mode = "rows"  # rows | companies
        filter_company: Optional[str] = None
        sort_by_fees = False
        sort_by_risk = 0  # 0 none, 1 asc, -1 desc
        filter_fond: Optional[str] = None

        def prompt_fond_choice(fonds: List[str]) -> Optional[str]:
            if not fonds:
                return None
            fonds = ["All"] + fonds
            prompt_h = min(len(fonds) + 4, curses.LINES - 2)
            prompt_w = min(max(len(f) for f in fonds) + 6, curses.COLS - 2)
            start_y = max(1, (curses.LINES - prompt_h) // 2)
            start_x = max(1, (curses.COLS - prompt_w) // 2)
            win = curses.newwin(prompt_h, prompt_w, start_y, start_x)
            win.keypad(True)
            win.box()
            win.addstr(0, 2, " Choix du fond ")
            idx = 0
            while True:
                for i, fval in enumerate(fonds[: prompt_h - 3]):
                    attr = curses.A_REVERSE if i == idx else 0
                    win.addstr(1 + i, 2, fval[: prompt_w - 4], attr)
                win.addstr(prompt_h - 2, 2, "Enter: valider / q/ESC: annuler")
                win.refresh()
                ch = win.getch()
                if ch in (curses.KEY_UP, ord("k")):
                    idx = max(0, idx - 1)
                elif ch in (curses.KEY_DOWN, ord("j")):
                    idx = min(len(fonds) - 1, idx + 1)
                elif ch in (ord("\n"), 10, 13):
                    choice = fonds[idx]
                    if choice == "All":
                        choice = None
                    del win
                    return choice
                elif ch in (ord("q"), ord("Q"), 27):
                    del win
                    return None

        def prompt_fond_choice_for_add(fonds: List[str]) -> Optional[str]:
            options = list(dict.fromkeys(fonds))
            options.append("<Nouveau>")
            prompt_h = min(len(options) + 4, curses.LINES - 2)
            prompt_w = min(max(len(f) for f in options) + 6, curses.COLS - 2)
            start_y = max(1, (curses.LINES - prompt_h) // 2)
            start_x = max(1, (curses.COLS - prompt_w) // 2)
            win = curses.newwin(prompt_h, prompt_w, start_y, start_x)
            win.keypad(True)
            win.box()
            win.addstr(0, 2, " Choisir le fond ")
            idx = 0
            while True:
                for i, fval in enumerate(options[: prompt_h - 3]):
                    attr = curses.A_REVERSE if i == idx else 0
                    win.addstr(1 + i, 2, fval[: prompt_w - 4], attr)
                win.addstr(prompt_h - 2, 2, "Enter: valider / q/ESC: annuler")
                win.refresh()
                ch = win.getch()
                if ch in (curses.KEY_UP, ord("k")):
                    idx = max(0, idx - 1)
                elif ch in (curses.KEY_DOWN, ord("j")):
                    idx = min(len(options) - 1, idx + 1)
                elif ch in (ord("q"), ord("Q"), 27):
                    del win
                    return None
                elif ch in (10, 13):
                    choice = options[idx]
                    del win
                    return choice

        def render():
            stdscr.erase()
            stdscr.addstr(0, 1, "Pipeline Expert")
            if mode == "companies":
                companies = sorted({row.get("management_company", "") for row in rows_list})
                draw_row(2, ["Sociétés de gestion", "", "", "", "", "", "", ""])
                y = 3
                visible_height = max(1, curses.LINES - 6)
                end_index = min(len(companies), top_offset + visible_height)
                for idx in range(top_offset, end_index):
                    name = companies[idx] or "(vide)"
                    draw_row(y, [name, "", "", "", "", "", "", ""], is_selected=(idx == selected))
                    y += 1
                if end_index < len(companies):
                    stdscr.addstr(curses.LINES - 2, 1, "... (scroll down for more)")
                stdscr.addstr(
                    curses.LINES - 1,
                    1,
                    "Tab: Soc. gestion | 1: Fond | 2: +produit | 3: -produit | r: tri SRI | f: tri frais | 0: rapport | d: ouvrir PDF | s: ISIN | Enter: expert | ESC/q: quitter",
                )
            else:
                current_rows = rows_list
                if filter_fond:
                    current_rows = [r for r in current_rows if str(r.get("fond", "")) == filter_fond]
                if filter_company:
                    current_rows = [r for r in current_rows if r.get("management_company", "") == filter_company]
                current_rows = sorted(
                    current_rows,
                    key=lambda r: float(r["frais_courants_pct"]) if str(r.get("frais_courants_pct", "")).replace(".", "", 1).isdigit() else float("inf"),
                ) if sort_by_fees else current_rows
                if sort_by_risk != 0:
                    current_rows = sorted(
                        current_rows,
                        key=lambda r: int(r["sri"]) if str(r.get("sri", "")).isdigit() else (999 if sort_by_risk == 1 else -999),
                        reverse=True if sort_by_risk == -1 else False,
                    )
                draw_row(2, headers)
                y = 3
                visible_height = max(1, curses.LINES - 10)  # leave space for hint and padding
                end_index = min(len(current_rows), top_offset + visible_height)
                for idx in range(top_offset, end_index):
                    row = current_rows[idx]
                    sri_str = str(row.get("sri", ""))
                    draw_row(
                        y,
                        [
                            row.get("fond", ""),
                            row.get("product_name", ""),
                            row.get("isin", ""),
                            sri_str,
                            row.get("horizon", ""),
                            row.get("frais_courants_pct", ""),
                            row.get("frais_entree_pct", ""),
                            row.get("frais_sortie_pct", ""),
                        ],
                        highlight_risk=sri_str,
                        is_selected=(idx == selected),
                    )
                    y += 1
                if end_index < len(current_rows):
                    stdscr.addstr(curses.LINES - 2, 1, "... (scroll down for more)")
                stdscr.addstr(
                    curses.LINES - 1,
                    1,
                    "Tab: Soc. gestion | 1: Fond | 2: +produit | 3: -produit | r: tri SRI | f: tri frais | 0: rapport | d: ouvrir PDF | s: ISIN | Enter: expert | ESC/q: quitter",
                )
            stdscr.refresh()

        def show_popup(row: Dict[str, str]):
            popup_lines = [
                ("Asset class :", row.get("asset_class", "")),
                ("Région      :", row.get("investment_region", "")),
                ("Style       :", row.get("management_style", "")),
                ("SFDR        :", row.get("sfdr_classification", "")),
                ("Devise      :", row.get("currency", "")),
                ("Soc. gestion:", row.get("management_company", "")),
                ("Objectif    :", row.get("objective_summary", "")),
                ("Benchmark   :", row.get("benchmark", "")),
                ("Risques     :", row.get("main_risks", "")),
                ("Nav freq    :", row.get("nav_frequency", "")),
                ("Liquidité   :", row.get("liquidity_constraints", "")),
                ("Perf fee %  :", row.get("performance_fee_pct", "")),
                ("Mgmt fee %  :", row.get("management_fees_pct", "")),
                ("Tx costs %  :", row.get("transaction_costs_pct", "")),
                ("Autres %    :", row.get("other_costs_pct", "")),
            ]
            width = min(curses.COLS - 2, max(len(lbl) + 1 + len(str(val)) for lbl, val in popup_lines) + 4)
            content_width = max(20, width - 4)
            wrapped = []
            for label, val in popup_lines:
                available = max(8, content_width - len(label) - 1)
                chunks = textwrap.wrap(str(val), width=available, replace_whitespace=False, drop_whitespace=False) or [""]
                for idx, chunk in enumerate(chunks):
                    wrapped.append((label if idx == 0 else " " * len(label), chunk))
            height = min(curses.LINES - 2, max(10, len(wrapped) + 4))
            start_y = max(1, (curses.LINES - height) // 2)
            start_x = max(1, (curses.COLS - width) // 2)
            win = curses.newwin(height, width, start_y, start_x)
            win.box()
            win.addstr(0, 2, " Expert ")
            for i, (label, val) in enumerate(wrapped, start=1):
                if i >= height - 1:
                    break
                win.addstr(i, 2, label, curses.color_pair(1))
                win.addstr(i, 3 + len(label), f" {val}"[: content_width - len(label) - 1])
            win.addstr(height - 2, 2, "0: rapport | ESC/q: fermer")
            win.refresh()
            while True:
                ch = win.getch()
                if ch == ord("0"):
                    report = _run_report_with_progress(row)
                    _show_report_popup(report, color_pair=5)
                    win.touchwin()
                    win.refresh()
                elif ch in (ord("q"), ord("Q"), 27):  # 27 = ESC
                    break
            del win

        def _run_report_with_progress(row: Dict[str, str]) -> str:
            report_text = "Analyse en cours..."
            h = 5
            w = 60
            start_y = max(1, (curses.LINES - h) // 2)
            start_x = max(1, (curses.COLS - w) // 2)
            win = curses.newwin(h, w, start_y, start_x)
            win.bkgd(" ", curses.color_pair(5))
            win.box()
            stop_event = threading.Event()

            def worker():
                nonlocal report_text
                report_text = _analyst_report(row.get("isin", ""), row.get("management_company", ""), debug=debug)
                stop_event.set()

            t = threading.Thread(target=worker, daemon=True)
            t.start()
            spinner = ["|", "/", "-", "\\"]
            idx = 0
            while not stop_event.is_set():
                win.erase()
                win.bkgd(" ", curses.color_pair(5))
                win.box()
                msg = f"Analyse de reporting en cours... {spinner[idx % len(spinner)]}"
                win.addstr(2, 2, msg[: w - 4])
                win.refresh()
                idx += 1
                time.sleep(0.1)
            t.join()
            del win
            # If reporting missing, offer URL download after user confirms
            if report_text.startswith("Reporting introuvable"):
                confirm = prompt_confirmation(
                    "Téléchargement manuel",
                    f"{report_text}\nAppuyez sur 'u' pour saisir une URL ou ESC pour annuler.",
                )
                if confirm:
                    url = prompt_report_url()
                    if url:
                        dest = ANALYST_REPORT_DIR / f"reporting-{row.get('isin', '')}.pdf"
                        success = _download_url_to(dest, url)
                        if success:
                            report_text = (
                                f"Reporting téléchargé dans {dest}.\n"
                                "Relancez l'analyse (touche 0) pour générer le rapport."
                            )
            return report_text

        def prompt_confirmation(title: str, message: str) -> bool:
            lines = message.splitlines()
            max_len = max(len(line) for line in lines) if lines else 0
            h = min(curses.LINES - 2, max(5, len(lines) + 3))
            w = min(curses.COLS - 2, max(30, max_len + 4))
            start_y = max(1, (curses.LINES - h) // 2)
            start_x = max(1, (curses.COLS - w) // 2)
            win = curses.newwin(h, w, start_y, start_x)
            win.bkgd(" ", curses.color_pair(5))
            win.box()
            win.addstr(0, 2, f" {title} ")
            for idx, line in enumerate(lines[: h - 2]):
                win.addstr(1 + idx, 2, line[: w - 4])
            win.refresh()
            while True:
                ch = win.getch()
                if ch in (ord("u"), ord("U")):
                    del win
                    return True
                if ch in (27, ord("q"), ord("Q")):
                    del win
                    return False

        def prompt_report_url() -> Optional[str]:
            prompt_h = 5
            # Allow long URLs (e.g., with query strings); cap by terminal width
            prompt_w = min(140, curses.COLS - 2)
            start_y = max(1, (curses.LINES - prompt_h) // 2 + 3)
            start_x = max(1, (curses.COLS - prompt_w) // 2)
            win = curses.newwin(prompt_h, prompt_w, start_y, start_x)
            curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_RED)
            win.bkgd(" ", curses.color_pair(6))
            win.box()
            win.addstr(0, 2, " Entrer l'URL du reporting ")
            win.addstr(2, 2, "URL: ")
            curses.curs_set(1)
            win.refresh()
            buffer = ""
            while True:
                ch = win.getch()
                if ch in (10, 13):  # Enter
                    break
                if ch in (27,):  # ESC
                    buffer = ""
                    break
                if ch in (curses.KEY_BACKSPACE, 127, 8):
                    buffer = buffer[:-1]
                elif 32 <= ch <= 126 and len(buffer) < 512:
                    buffer += chr(ch)
                # Display only the tail that fits and keep cursor inside window bounds
                display_room = max(1, prompt_w - 9)
                to_show = buffer[-display_room:]
                win.addstr(2, 7, " " * display_room)
                win.addstr(2, 7, to_show)
                cursor_x = min(prompt_w - 2, 7 + len(to_show))
                try:
                    win.move(2, cursor_x)
                except curses.error:
                    pass
                win.refresh()
            curses.curs_set(0)
            del win
            return buffer.strip() if buffer.strip() else None

        def prompt_isin() -> Optional[str]:
            prompt_h = 5
            prompt_w = 40
            start_y = max(1, (curses.LINES - prompt_h) // 2)
            start_x = max(1, (curses.COLS - prompt_w) // 2)
            win = curses.newwin(prompt_h, prompt_w, start_y, start_x)
            win.box()
            win.addstr(0, 2, " Recherche ISIN ")
            win.addstr(2, 2, "ISIN: ")
            curses.curs_set(1)
            win.refresh()
            buffer = ""
            while True:
                ch = win.getch()
                if ch in (10, 13):  # Enter
                    break
                if ch in (27,):  # ESC
                    buffer = ""
                    break
                if ch in (curses.KEY_BACKSPACE, 127, 8):
                    buffer = buffer[:-1]
                elif 32 <= ch <= 126 and len(buffer) < 20:
                    buffer += chr(ch)
                win.addstr(2, 8, " " * (prompt_w - 10))
                win.addstr(2, 8, buffer.upper())
                win.move(2, 8 + len(buffer))
                win.refresh()
            curses.curs_set(0)
            del win
            return buffer.strip().upper() if buffer.strip() else None

        def prompt_fond_name() -> Optional[str]:
            prompt_h = 5
            prompt_w = 50
            start_y = max(1, (curses.LINES - prompt_h) // 2)
            start_x = max(1, (curses.COLS - prompt_w) // 2)
            win = curses.newwin(prompt_h, prompt_w, start_y, start_x)
            win.box()
            win.addstr(0, 2, " Nouveau fond ")
            win.addstr(2, 2, "Nom: ")
            curses.curs_set(1)
            win.refresh()
            buffer = ""
            while True:
                ch = win.getch()
                if ch in (10, 13):
                    break
                if ch in (27,):
                    buffer = ""
                    break
                if ch in (curses.KEY_BACKSPACE, 127, 8):
                    buffer = buffer[:-1]
                elif 32 <= ch <= 126 and len(buffer) < 40:
                    buffer += chr(ch)
                win.addstr(2, 7, " " * (prompt_w - 9))
                win.addstr(2, 7, buffer)
                win.move(2, min(prompt_w - 2, 7 + len(buffer)))
                win.refresh()
            curses.curs_set(0)
            del win
            return buffer.strip() if buffer.strip() else None

        def prompt_delete_confirmation(row: Dict[str, str]) -> Optional[bool]:
            product = row.get("product_name", "") or "(sans nom)"
            isin_val = row.get("isin", "") or "(ISIN?)"
            msg_lines = [
                f"Supprimer '{product}'",
                f"ISIN: {isin_val}",
                "Supprimer aussi les fichiers PDF/TXT associés ?",
                "[o] Oui (avec fichiers)  |  [n] Oui (CSV seulement)  |  ESC annuler",
            ]
            h = min(curses.LINES - 2, max(7, len(msg_lines) + 3))
            w = min(curses.COLS - 2, max(40, max(len(m) for m in msg_lines) + 4))
            start_y = max(1, (curses.LINES - h) // 2)
            start_x = max(1, (curses.COLS - w) // 2)
            win = curses.newwin(h, w, start_y, start_x)
            win.bkgd(" ", curses.color_pair(4))
            win.box()
            win.addstr(0, 2, " Confirmation suppression ")
            for i, line in enumerate(msg_lines[: h - 2]):
                win.addstr(1 + i, 2, line[: w - 4])
            win.refresh()
            while True:
                ch = win.getch()
                if ch in (ord("o"), ord("O")):
                    del win
                    return True  # delete files too
                if ch in (ord("n"), ord("N")):
                    del win
                    return False  # CSV only
                if ch in (27, ord("q"), ord("Q")):
                    del win
                    return None

        def show_message(title: str, message: str, color_pair: int = 0, height: int = 6) -> None:
            lines = message.splitlines()
            h = min(curses.LINES - 2, max(height, len(lines) + 4))
            w = min(curses.COLS - 2, max(30, max((len(l) for l in lines), default=0) + 4))
            start_y = max(1, (curses.LINES - h) // 2)
            start_x = max(1, (curses.COLS - w) // 2)
            win = curses.newwin(h, w, start_y, start_x)
            if color_pair:
                win.bkgd(" ", curses.color_pair(color_pair))
            win.box()
            win.addstr(0, 2, f" {title} ")
            for i, line in enumerate(lines[: h - 3]):
                win.addstr(1 + i, 2, line[: w - 4])
            win.addstr(h - 2, 2, "Enter/ESC pour fermer")
            win.refresh()
            while True:
                ch = win.getch()
                if ch in (10, 13, 27, ord("q"), ord("Q")):
                    break
            del win

        def delete_product(row: Dict[str, str], delete_files: bool) -> None:
            fond_name = str(row.get("fond", "default")) or "default"
            isin_val = row.get("isin", "")
            source_pdf = row.get("source_pdf", "")
            if fond_name == "default":
                target_csv = OUTPUT_CSV
            else:
                target_csv = Path("data") / fond_name / f"dic_summary_expert_{fond_name}.csv"
            def _filter_csv(csv_path: Path):
                if not csv_path.exists():
                    return
                try:
                    with open(csv_path, newline="") as f:
                        reader = list(csv.DictReader(f))
                    kept = [
                        r for r in reader
                        if not (
                            (isin_val and str(r.get("isin", "")).upper() == isin_val.upper())
                            and (not source_pdf or Path(r.get("source_pdf", "")).name == Path(source_pdf).name)
                        )
                    ]
                    with open(csv_path, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                        writer.writeheader()
                        writer.writerows(kept)
                except Exception as exc:
                    logging.error("Erreur lors de la mise à jour CSV %s: %s", csv_path, exc)
            _filter_csv(target_csv)
            _filter_csv(GLOBAL_CSV)
            if delete_files:
                try:
                    if source_pdf:
                        Path(source_pdf).unlink(missing_ok=True)
                except Exception as exc:
                    logging.error("Impossible de supprimer le PDF %s: %s", source_pdf, exc)
                try:
                    if isin_val:
                        (ANALYST_REPORT_DIR / f"rapport-{isin_val}.txt").unlink(missing_ok=True)
                        (ANALYST_REPORT_DIR / f"reporting-{isin_val}.pdf").unlink(missing_ok=True)
                except Exception as exc:
                    logging.error("Impossible de supprimer les rapports pour %s: %s", isin_val, exc)
            # mise à jour en mémoire
            nonlocal rows_list
            rows_list = [
                r for r in rows_list
                if not (
                    (isin_val and str(r.get("isin", "")).upper() == isin_val.upper())
                    and (not source_pdf or Path(r.get("source_pdf", "")).name == Path(source_pdf).name)
                )
            ]
            # regénère global pour cohérence
            regenerate_global_csv()

        def prompt_parse_next() -> bool:
            message = "Téléchargement terminé.\nAppuyez sur 'p' pour lancer le parsing ou ESC pour annuler."
            lines = message.splitlines()
            h = min(curses.LINES - 2, max(6, len(lines) + 4))
            w = min(curses.COLS - 2, max(40, max((len(l) for l in lines), default=0) + 4))
            start_y = max(1, (curses.LINES - h) // 2)
            start_x = max(1, (curses.COLS - w) // 2)
            win = curses.newwin(h, w, start_y, start_x)
            win.box()
            win.addstr(0, 2, " Étape suivante ")
            for i, line in enumerate(lines[: h - 3]):
                win.addstr(1 + i, 2, line[: w - 4])
            win.addstr(h - 2, 2, "p: parser | ESC: annuler")
            win.refresh()
            while True:
                ch = win.getch()
                if ch in (ord("p"), ord("P")):
                    del win
                    return True
                if ch in (27, ord("q"), ord("Q")):
                    del win
                    return False

        def parse_with_spinner(pdf_path: Path) -> Optional[Dict[str, str]]:
            result: Dict[str, str] = {}
            h = 5
            w = 60
            start_y = max(1, (curses.LINES - h) // 2)
            start_x = max(1, (curses.COLS - w) // 2)
            win = curses.newwin(h, w, start_y, start_x)
            win.bkgd(" ", curses.color_pair(5))
            win.box()
            spinner = ["|", "/", "-", "\\"]
            stop_event = threading.Event()

            def worker():
                nonlocal result
                try:
                    result = parse_pdf_fields(pdf_path, debug=debug)
                except Exception as exc:
                    result = {"error": str(exc)}
                finally:
                    stop_event.set()

            t = threading.Thread(target=worker, daemon=True)
            t.start()
            idx = 0
            while not stop_event.is_set():
                win.erase()
                win.bkgd(" ", curses.color_pair(5))
                win.box()
                msg = f"Parsing en cours... {spinner[idx % len(spinner)]}"
                win.addstr(2, 2, msg[: w - 4])
                win.refresh()
                idx += 1
                time.sleep(0.1)
            t.join()
            del win
            if "error" in result:
                show_message("Erreur", f"Parsing échoué:\n{result['error']}", color_pair=4)
                return None
            return result

        def add_product_flow():
            fonds_available = sorted({str(r.get("fond", "")) or "default" for r in rows_list})
            fond_choice = prompt_fond_choice_for_add(fonds_available)
            if not fond_choice:
                return
            if fond_choice == "<Nouveau>":
                fond_choice = prompt_fond_name()
                if not fond_choice:
                    return
            isin_val = prompt_isin()
            if not isin_val:
                return
            url = prompt_report_url()
            if not url:
                return
            if fond_choice == "default":
                fond_dir = DATA_DIR
                output_csv = OUTPUT_CSV
            else:
                fond_dir = Path("data") / fond_choice
                output_csv = fond_dir / f"dic_summary_expert_{fond_choice}.csv"
            fond_dir.mkdir(parents=True, exist_ok=True)
            dest_pdf = fond_dir / f"{isin_val}.pdf"
            success = _download_url_to(dest_pdf, url)
            if not success:
                show_message("Téléchargement échoué", "Impossible de télécharger le PDF.\nVérifiez l'URL et réessayez.", color_pair=4)
                return
            if not prompt_parse_next():
                return
            fields = parse_with_spinner(dest_pdf)
            if not fields:
                return
            fields["fond"] = fond_choice
            fields["source_pdf"] = str(dest_pdf)
            write_output([fields], output_csv=output_csv, fond_label=fond_choice)
            rows_list.append(fields)
            show_message("Succès", f"Produit ajouté au fond {fond_choice}.", color_pair=1)

        def _clean_report_text(text: str) -> str:
            cleaned = text.replace("```", "")
            cleaned = cleaned.replace("**", "")
            cleaned = re.sub(r"\s+\|\s+", " | ", cleaned)
            cleaned = re.sub(r"[ \t]+", " ", cleaned)
            return cleaned.strip()

        def _show_report_popup(text: str, color_pair: int = 0):
            cleaned = _clean_report_text(text)
            paragraphs = cleaned.splitlines()
            wrapped: List[str] = []
            width_avail = max(30, curses.COLS - 6)
            for para in paragraphs:
                para = para.strip()
                if not para:
                    wrapped.append("")
                    continue
                wrapped.extend(textwrap.wrap(para, width=width_avail, replace_whitespace=False))
            lines = wrapped or [cleaned]
            height = min(curses.LINES - 2, max(6, len(lines) + 4))
            width = min(curses.COLS - 2, max(30, max(len(l) for l in lines) + 4))
            start_y = max(1, (curses.LINES - height) // 2)
            start_x = max(1, (curses.COLS - width) // 2)
            win = curses.newwin(height, width, start_y, start_x)
            win.keypad(True)
            if color_pair:
                win.bkgd(" ", curses.color_pair(color_pair))
            win.box()
            win.addstr(0, 2, " Rapport ")
            scroll = 0

            def render_body():
                win.erase()
                if color_pair:
                    win.bkgd(" ", curses.color_pair(color_pair))
                win.box()
                win.addstr(0, 2, " Rapport ")
                visible_height = height - 3
                for i in range(visible_height):
                    idx = scroll + i
                    if idx >= len(lines):
                        break
                    win.addstr(1 + i, 2, lines[idx][: width - 4])
                win.addstr(height - 2, 2, "↑/↓ pour défiler, Enter/ESC/q pour fermer")
                win.refresh()

            render_body()
            while True:
                ch = win.getch()
                if ch in (ord("\n"), ord("q"), ord("Q"), 27):
                    break
                elif ch in (curses.KEY_UP, ord("k")):
                    scroll = max(0, scroll - 1)
                    render_body()
                elif ch in (curses.KEY_DOWN, ord("j")):
                    if scroll + (height - 3) < len(lines):
                        scroll += 1
                        render_body()
            del win

        render()
        while True:
            ch = stdscr.getch()
            if ch in (ord("q"), ord("Q")):
                break
            if ch == 9:  # Tab
                if mode == "rows":
                    mode = "companies"
                else:
                    mode = "rows"
                selected = 0
                top_offset = 0
                render()
                continue
            if ch == 27:  # ESC clears filter and returns to rows
                filter_company = None
                filter_fond = None
                mode = "rows"
                selected = 0
                top_offset = 0
                sort_by_fees = False
                sort_by_risk = 0
                render()
                continue
            if ch in (ord("r"), ord("R")):
                sort_by_risk = 1 if sort_by_risk != 1 else -1
                sort_by_fees = False
                selected = 0
                top_offset = 0
                render()
                continue
            if ch in (ord("f"), ord("F")):
                sort_by_fees = not sort_by_fees
                sort_by_risk = 0
                selected = 0
                top_offset = 0
                render()
                continue

            if mode == "companies":
                companies = sorted({row.get("management_company", "") for row in rows_list})
                if ch in (curses.KEY_UP, ord("k")):
                    selected = max(0, selected - 1)
                    if selected < top_offset:
                        top_offset = selected
                    render()
                elif ch in (curses.KEY_DOWN, ord("j")):
                    selected = min(len(companies) - 1, selected + 1)
                    visible_height = max(1, curses.LINES - 6)
                    if selected >= top_offset + visible_height:
                        top_offset = max(0, selected - visible_height + 1)
                    render()
                elif ch in (ord("\n"), curses.KEY_ENTER, 10, 13):
                    if companies:
                        filter_company = companies[selected]
                    mode = "rows"
                    selected = 0
                    top_offset = 0
                    render()
                continue

            # rows mode
            current_rows = rows_list
            if filter_fond:
                current_rows = [r for r in current_rows if str(r.get("fond", "")) == filter_fond]
            if filter_company:
                current_rows = [r for r in current_rows if r.get("management_company", "") == filter_company]
            if ch in (curses.KEY_UP, ord("k")):
                selected = max(0, selected - 1)
                if selected < top_offset:
                    top_offset = selected
                render()
            elif ch in (curses.KEY_DOWN, ord("j")):
                selected = min(len(current_rows) - 1, selected + 1)
                visible_height = max(1, curses.LINES - 10)
                if selected >= top_offset + visible_height:
                    top_offset = max(0, selected - visible_height + 1)
                render()
            elif ch in (ord("\n"), curses.KEY_ENTER, 10, 13):
                if current_rows:
                    show_popup(current_rows[selected])
                render()
            elif ch in (ord("s"), ord("S")):
                isin_query = prompt_isin()
                if isin_query:
                    target = next((r for r in rows_list if str(r.get("isin", "")).upper() == isin_query), None)
                    if target:
                        show_popup(target)
                render()
            elif ch in (ord("d"), ord("D")):
                if current_rows:
                    pdf_path = Path(current_rows[selected].get("source_pdf", ""))
                    try:
                        subprocess.Popen(["open", str(pdf_path.resolve())])
                    except Exception as exc:
                        logging.error("Cannot open PDF %s: %s", pdf_path, exc)
                render()
            elif ch in (ord("1"),):
                fonds = sorted({str(r.get("fond", "")) or "default" for r in rows_list})
                choice = prompt_fond_choice(fonds)
                filter_fond = choice  # None clears the filter when "All" is chosen
                selected = 0
                top_offset = 0
                render()
            elif ch in (ord("2"),):
                add_product_flow()
                render()
            elif ch in (ord("3"),):
                if current_rows:
                    row = current_rows[selected]
                    choice = prompt_delete_confirmation(row)
                    if choice is not None:
                        delete_product(row, delete_files=choice)
                        selected = min(selected, max(0, len(rows_list) - 1))
                    render()

    curses.wrapper(_inner)


def clean_analyst_artifacts() -> None:
    try:
        if ASSISTANT_ID_FILE.exists():
            ASSISTANT_ID_FILE.unlink()
    except Exception as exc:
        logging.error("Impossible de supprimer %s: %s", ASSISTANT_ID_FILE, exc)
    try:
        for path in ANALYST_REPORT_DIR.glob("rapport-*.txt"):
            path.unlink()
    except Exception as exc:
        logging.error("Impossible de supprimer les rapports analyste: %s", exc)


def run_pipeline(
    source_url: str,
    skip_ui: bool = False,
    do_download: bool = False,
    do_parse: bool = False,
    parse_limit: Optional[int] = None,
    no_parse_ui: bool = True,
    debug: bool = False,
    single_pdf: Optional[str] = None,
    report_isin: Optional[str] = None,
    fond_suffix: Optional[str] = None,
    regen_global: bool = False,
    clean_reports: bool = False,
) -> None:
    if regen_global:
        regenerate_global_csv()
        return
    if clean_reports:
        clean_analyst_artifacts()
        logging.info("Nettoyage des artefacts analyste terminé.")
        return
    data_dir = DATA_DIR
    output_csv = OUTPUT_CSV
    if fond_suffix:
        data_dir = Path("data") / fond_suffix
        output_csv = data_dir / f"dic_summary_expert_{fond_suffix}.csv"
        data_dir.mkdir(parents=True, exist_ok=True)
    display_csv = output_csv if fond_suffix else GLOBAL_CSV
    if report_isin:
        report = _analyst_report(report_isin, debug=debug)
        print(report)
        return
    pdf_paths_override = None
    if single_pdf:
        pdf_path_obj = Path(single_pdf)
        if not pdf_path_obj.exists():
            logging.error("Specified PDF %s does not exist", single_pdf)
            return
        pdf_paths_override = [pdf_path_obj]
        logging.info("Parsing single PDF: %s", pdf_path_obj)
        do_download = False

    if do_download and not pdf_paths_override:
        pdf_links = fetch_pdf_links(source_url)
        logging.info("Found %d pdf links", len(pdf_links))
        download_pdfs(pdf_links, data_dir=data_dir)
    elif not pdf_paths_override:
        logging.info("Download skipped (use --download to enable); processing existing PDFs in %s", data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
    if no_parse_ui:
        rows = load_output_rows(output_csv=display_csv)
        if not rows:
            logging.info("No rows to display; ensure %s exists by running with --parse first.", display_csv)
            return
        if not skip_ui:
            display_with_curses(rows)
        return

    if not do_parse:
        logging.info("Parsing skipped (use --parse to enable).")
        return

    rows = process_all_pdfs(limit=parse_limit, debug=debug, pdf_paths=pdf_paths_override, data_dir=data_dir, output_csv=output_csv)
    write_output(rows, output_csv=output_csv, fond_label=fond_suffix or "default")
    logging.info("Wrote %s with %d new rows", output_csv, len(rows))
    if not skip_ui:
        display_with_curses(load_output_rows(output_csv=display_csv), debug=debug)


def _parse_num(val: str) -> Optional[int]:
    if isinstance(val, int):
        return val
    if str(val).lower() == "all":
        return None
    try:
        return int(val)
    except Exception:
        logging.warning("Invalid --num value %s, falling back to 3", val)
        return 3


def main():
    parser = argparse.ArgumentParser(description="Expert DIC/KID collector and extractor (OpenAI-only)")
    parser.add_argument("--url", default=DEFAULT_URL, help="Page URL listing the DIC PDFs")
    parser.add_argument("--no-ui", action="store_true", help="Skip curses display")
    parser.add_argument("--download", action="store_true", help="Fetch and download PDFs from the URL before parsing")
    parser.add_argument("--parse", action="store_true", help="Parse existing/downloaded PDFs and generate outputs")
    parser.add_argument("--file", help="Parse a single PDF file (path) instead of all in data/dic_pdfs")
    parser.add_argument(
        "--num",
        default="3",
        help='Limit number of PDFs to parse (integer or "all"). Default: 3',
    )
    parser.add_argument("--debug", action="store_true", help="Log raw OpenAI JSON responses per file")
    parser.add_argument("--rapport", help="Générer uniquement un rapport analyste pour l'ISIN fourni")
    parser.add_argument("--fond", help="Nom complet du sous-répertoire dans data (ex: data/monfond) pour parser/écrire dic_summary_expert_<nom>.csv")
    parser.add_argument("--regen-global", action="store_true", help="Regénérer le CSV global data/dic_summary_expert_all.csv à partir des CSV existants")
    parser.add_argument("--clean", action="store_true", help="Supprimer assistant_id.txt et rapport-*.txt dans data/analyste")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    run_pipeline(
        args.url,
        skip_ui=args.no_ui,
        do_download=args.download,
        do_parse=args.parse,
        parse_limit=_parse_num(args.num),
        no_parse_ui=not args.parse,
        debug=args.debug,
        single_pdf=args.file,
        report_isin=args.rapport,
        fond_suffix=args.fond,
        regen_global=args.regen_global,
        clean_reports=args.clean,
    )


if __name__ == "__main__":
    main()
