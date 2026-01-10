import argparse
import csv
import json
import logging
import os
import re
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
OUTPUT_CSV = Path("data/dic_summary.csv")
DOWNLOAD_LOG = Path("data/download_log.csv")
DEFAULT_URL = "https://priips.predica.com/credit-agricole/consultation-support"
MISSING = "MISSING"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")


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


def download_pdfs(urls: List[str]) -> List[Dict[str, str]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log_rows = []
    for url in urls:
        filename = _safe_filename(url)
        dest = DATA_DIR / filename
        downloaded = False
        try:
            if dest.exists():
                logging.info("Skip existing %s", dest)
            else:
                with requests.get(url, stream=True, timeout=30) as resp:
                    resp.raise_for_status()
                    with open(dest, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                downloaded = True
                logging.info("Downloaded %s", dest)
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
            # layout=True retains spacing similar to pdfplumber CLI --format text
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


def _find_sri(text: str) -> str:
    patterns = [
        r"Indice [^0-9]*([1-7])\s*/\s*7",
        r"SRI[^0-9]*([1-7])",
        r"([1-7])\s*/\s*7",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    return MISSING


def _find_horizon(text: str) -> str:
    m = re.search(r"Horizon de placement recommand[\u00e9e]\s*:?\s*(.+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return MISSING


def _find_percentage(text: str, label: str) -> str:
    m = re.search(rf"{label}.*?(\d+[.,]\d+)\s*%", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).replace(",", ".")
    return MISSING


def _find_product_name(text: str) -> str:
    m = re.search(r"Nom du produit\s*:?\s*(.+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0][:120] if lines else MISSING


def parse_pdf_fields_with_openai(pdf_path: Path, debug: bool = False) -> Dict[str, str]:
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return _empty_fields(pdf_path)

    client = OpenAI()
    prompt = textwrap.dedent(
        f"""
        Tu es une IA spécialisée dans l’extraction d’informations structurées à partir de documents financiers, en particulier les Documents d’Informations Clés (DIC / KID PRIIPs) en français.

On va te fournir le TEXTE BRUT extrait d’un DIC (sans mise en page PDF).

Ta tâche est d’identifier les informations pertinentes et de produire un objet JSON avec EXACTEMENT ces clés :

- "product_name"
- "isin"
- "sri"
- "horizon"
- "frais_courants_pct"
- "frais_entree_pct"
- "frais_sortie_pct"

Tu dois analyser tout le document, y compris les sections “Produit”, “Indicateur de risque”, “Coûts” et “Période de détention recommandée”.

Tu dois répondre UNIQUEMENT avec un JSON valide, sans texte supplémentaire, sans explication.

----------------------------------------------------
RÈGLES D’EXTRACTION
----------------------------------------------------

1) product_name
Repère le nom exact dans la section Produit ou Nom du produit.
Ne modifie rien (pas de reformulation). Si non trouvé → null.

2) isin
Repère “ISIN” ou “Code ISIN”.
Extrais le premier code ISIN à 12 caractères. Si absent → null.

3) sri
Repère la phrase contenant “classe de risque X sur 7” ou “X sur une échelle de 1 à 7”.
Renvoie seulement le chiffre X (entier). Si ambigu → null.

4) horizon
Repère “Période de détention recommandée : … ans” ou “Horizon de placement recommandé”.
Renvoie le nombre d’années (entier). Si absent → null.

5) frais_courants_pct
Utilise en priorité “Frais courants : X %”.
Sinon additionne :
- frais de gestion et autres frais administratifs
- + coûts de transaction
Renvoie un nombre JSON (ex: 2.26), sans signe %. Si impossible → null.

6) frais_entree_pct
Repère coûts d’entrée :
- “X %” → renvoyer X
- “Nous ne facturons pas” ou “0 %” → 0
Sinon → null.

7) frais_sortie_pct
Même logique que frais d’entrée, pour les coûts de sortie.

----------------------------------------------------
FORMAT DE SORTIE OBLIGATOIRE
----------------------------------------------------

Tu dois renvoyer UNIQUEMENT un objet JSON valide, du type :

{{
  "product_name": "...",
  "isin": "...",
  "sri": 4,
  "horizon": 5,
  "frais_courants_pct": 2.26,
  "frais_entree_pct": 3.5,
  "frais_sortie_pct": 0
}}

        Texte du PDF:
        {text}
        """
    )
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = resp.choices[0].message.content or ""
        data = _safe_parse_json(content, pdf_path)
        if debug:
            logging.info("OpenAI raw response for %s: %s", pdf_path.name, content)
        if data is None:
            return _empty_fields(pdf_path)
        return {
            "product_name": data.get("product_name", MISSING),
            "isin": data.get("isin", MISSING),
            "sri": data.get("sri", MISSING),
            "horizon": data.get("horizon", MISSING),
            "frais_courants_pct": data.get("frais_courants_pct", MISSING),
            "frais_entree_pct": data.get("frais_entree_pct", MISSING),
            "frais_sortie_pct": data.get("frais_sortie_pct", MISSING),
            "source_pdf": str(pdf_path),
        }
    except Exception as exc:
        logging.error("OpenAI parsing failed for %s: %s", pdf_path, exc)
        return _empty_fields(pdf_path)


def _safe_parse_json(content: str, pdf_path: Path) -> Optional[Dict[str, str]]:
    """Try to parse JSON while handling code fences and stray text."""
    cleaned = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # fallback: extract substring between first { and last }
    brace_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if brace_match:
        snippet = brace_match.group(0)
        try:
            return json.loads(snippet)
        except Exception:
            logging.error("OpenAI parsing failed to parse JSON for %s; cleaned snippet: %s", pdf_path, snippet[:200])
            return None
    logging.error("OpenAI parsing failed to parse JSON for %s; raw: %s", pdf_path, content[:200])
    return None




def _empty_fields(pdf_path: Path) -> Dict[str, str]:
    return {
        "product_name": MISSING,
        "isin": MISSING,
        "sri": MISSING,
        "horizon": MISSING,
        "frais_courants_pct": MISSING,
        "frais_entree_pct": MISSING,
        "frais_sortie_pct": MISSING,
        "source_pdf": str(pdf_path),
    }


def parse_pdf_fields(pdf_path: Path, use_openai: bool = False, debug: bool = False) -> Dict[str, str]:
    text = extract_text_from_pdf(pdf_path)
    if use_openai:
        return parse_pdf_fields_with_openai(pdf_path, debug=debug)
    if not text:
        return _empty_fields(pdf_path)

    clean_text = text.replace("\u00a0", " ")
    return {
        "product_name": _find_product_name(clean_text),
        "isin": _find_isin(clean_text),
        "sri": _find_sri(clean_text),
        "horizon": _find_horizon(clean_text),
        "frais_courants_pct": _find_percentage(clean_text, "Frais courants"),
        "frais_entree_pct": _find_percentage(clean_text, r"Frais d['\u2019]?entr[\u00e9e]e"),
        "frais_sortie_pct": _find_percentage(clean_text, "Frais de sortie"),
        "source_pdf": str(pdf_path),
    }


def process_all_pdfs(use_openai: bool = False, limit: Optional[int] = None, debug: bool = False) -> List[Dict[str, str]]:
    rows = []
    existing = set()
    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing.add(Path(r.get("source_pdf", "")).name)
    pdf_paths = sorted(DATA_DIR.glob("*.pdf"))
    parsed_count = 0
    for pdf_path in pdf_paths:
        if pdf_path.name in existing:
            logging.info("Skipping %s (already in CSV)", pdf_path.name)
            continue
        if not _has_isin_on_first_page(pdf_path):
            logging.info("Skipping %s (no ISIN on first page)", pdf_path)
            continue
        fields = parse_pdf_fields(pdf_path, use_openai=use_openai, debug=debug)
        rows.append(fields)
        parsed_count += 1
        if limit is not None and parsed_count >= limit:
            break
    return rows


def load_output_rows() -> List[Dict[str, str]]:
    if not OUTPUT_CSV.exists():
        logging.warning("Output CSV %s not found", OUTPUT_CSV)
        return []
    with open(OUTPUT_CSV, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_output(rows: List[Dict[str, str]]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    exists = OUTPUT_CSV.exists()
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "product_name",
                "isin",
                "sri",
                "horizon",
                "frais_courants_pct",
                "frais_entree_pct",
                "frais_sortie_pct",
                "source_pdf",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _risk_color_pair(risk: str) -> int:
    try:
        value = int(risk)
    except ValueError:
        return 0
    if value <= 2:
        return 1
    if value <= 4:
        return 2
    if value <= 5:
        return 3
    return 4


def display_with_curses(rows: List[Dict[str, str]]) -> None:
    def _inner(stdscr):
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.curs_set(0)

        headers = ["Produit", "ISIN", "SRI", "Horizon", "Frais courants", "Entrée", "Sortie"]
        col_widths = [20, 14, 5, 25, 14, 10, 10]

        def draw_row(y: int, cols: List[str], highlight_risk: Optional[str] = None):
            x = 1
            color = curses.color_pair(_risk_color_pair(highlight_risk) if highlight_risk else 0)
            for idx, col in enumerate(cols):
                col_str = "" if col is None else str(col)
                cell = col_str[: col_widths[idx]].ljust(col_widths[idx])
                stdscr.addstr(y, x, cell, color)
                x += col_widths[idx] + 1

        stdscr.addstr(0, 1, "MVP1 - DIC Extracts (risk colored, higher is hotter)")
        draw_row(2, headers)
        y = 3
        for row in rows:
            sri_str = str(row["sri"])
            icon = "!" * min(3, int(sri_str)) if sri_str.isdigit() else "?"
            draw_row(
                y,
                [
                    f"{icon} {row['product_name']}",
                    row["isin"],
                    sri_str,
                    row["horizon"],
                    row["frais_courants_pct"],
                    row["frais_entree_pct"],
                    row["frais_sortie_pct"],
                ],
                highlight_risk=sri_str,
            )
            y += 1
            if y >= curses.LINES - 1:
                stdscr.addstr(y, 1, "... (output truncated)")
                break
        stdscr.refresh()
        stdscr.addstr(curses.LINES - 1, 1, "Appuyez sur 'q' pour quitter.")
        while True:
            ch = stdscr.getch()
            if ch in (ord("q"), ord("Q")):
                break

    curses.wrapper(_inner)


def run_pipeline(
    source_url: str,
    skip_ui: bool = False,
    do_download: bool = False,
    use_openai: bool = False,
    do_parse: bool = False,
    parse_limit: Optional[int] = None,
    no_parse_ui: bool = False,
    debug: bool = False,
) -> None:
    if do_download:
        pdf_links = fetch_pdf_links(source_url)
        logging.info("Found %d pdf links", len(pdf_links))
        download_pdfs(pdf_links)
    else:
        logging.info("Download skipped (use --download to enable); processing existing PDFs in %s", DATA_DIR)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    if no_parse_ui:
        rows = load_output_rows()
        if not rows:
            logging.info("No rows to display; ensure %s exists by running with --parse first.", OUTPUT_CSV)
            return
        if not skip_ui:
            display_with_curses(rows)
        return

    if not do_parse:
        logging.info("Parsing skipped (use --parse to enable).")
        return

    rows = process_all_pdfs(use_openai=use_openai, limit=parse_limit, debug=debug)
    write_output(rows)
    logging.info("Wrote %s with %d rows", OUTPUT_CSV, len(rows))
    if not skip_ui:
        display_with_curses(load_output_rows())


def main():
    parser = argparse.ArgumentParser(description="MVP1 DIC/KID collector and extractor")
    parser.add_argument("--url", default=DEFAULT_URL, help="Page URL listing the DIC PDFs")
    parser.add_argument("--no-ui", action="store_true", help="Skip curses display")
    parser.add_argument("--download", action="store_true", help="Fetch and download PDFs from the URL before parsing")
    parser.add_argument("--parse", action="store_true", help="Parse existing/downloaded PDFs and generate outputs")
    parser.add_argument("--noparse", action="store_true", help="Skip parsing and display existing CSV data in curses")
    parser.add_argument(
        "--num",
        default="3",
        help='Limit number of PDFs to parse (integer or "all"). Default: 3',
    )
    parser.add_argument("--use-openai", action="store_true", help="Parse with OpenAI instead of regex (requires OPENAI_API_KEY)")
    parser.add_argument("--debug", action="store_true", help="Log raw OpenAI JSON responses per file")
    args = parser.parse_args()

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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    run_pipeline(
        args.url,
        skip_ui=args.no_ui,
        do_download=args.download,
        use_openai=args.use_openai,
        do_parse=args.parse,
        parse_limit=_parse_num(args.num),
        no_parse_ui=args.noparse,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
