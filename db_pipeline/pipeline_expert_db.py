"""
Pipeline Expert (DB version) — Étape 2 : schéma + lecture Postgres.

Ce script fournit :
- Le SQL de création des tables Postgres (référentiel global + vues utilisateur).
- Un petit client en lecture qui interroge la base et affiche les lignes visibles
  par un utilisateur (global – masqués + favoris).

Utilisation rapide :
  # Afficher le SQL
  python db_pipeline/pipeline_expert_db.py --print-sql

  # Écrire le SQL dans un fichier
  python db_pipeline/pipeline_expert_db.py --write schema.sql

  # Lire la base et afficher les produits visibles pour l'utilisateur "demo"
  export DATABASE_URL=postgresql://user:pass@host:5432/dbname
  python db_pipeline/pipeline_expert_db.py --user demo --limit 20
"""

import argparse
import contextlib
import curses
import io
import json
import logging
import os
import re
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import psycopg
from db_pipeline.ai_service import (
    ANALYST_MANAGER_URLS,
    ANALYST_REPORT_DIR,
    ASSISTANT_ID_FILE,
    analyst_report,
    parse_pdf_to_payload,
)
from db_pipeline.db_service import add_to_portfolio, fetch_one_by_isin, fetch_rows, upsert_product
from db_pipeline.file_service import download_pdf
from db_pipeline import tendance_moyen_term as tmt


def _redact_sensitive(text: str) -> str:
    """Mask API keys in log lines."""
    if not text:
        return text
    # apikey=XXXX in query strings
    text = re.sub(r"apikey=([^&\\s]+)", r"apikey=REDACTED", text, flags=re.IGNORECASE)
    # Bearer tokens
    text = re.sub(r"Bearer\\s+[A-Za-z0-9._-]+", "Bearer REDACTED", text, flags=re.IGNORECASE)
    return text


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        return _redact_sensitive(msg)


def _setup_logging(debug: bool, log_file: Optional[str]) -> None:
    level = logging.DEBUG if debug else logging.INFO
    handler: logging.Handler
    if log_file:
        handler = logging.FileHandler(log_file, encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(RedactingFormatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.basicConfig(level=level, handlers=[handler], force=True)


CREATE_TABLES_SQL = """
-- Table principale : référentiel des produits/fonds (commune à tous)
CREATE TABLE IF NOT EXISTS products_global (
    id              SERIAL PRIMARY KEY,
    isin            TEXT UNIQUE NOT NULL,
    fond            TEXT DEFAULT 'default',
    product_name    TEXT,
    sri             INTEGER,
    horizon         TEXT,
    frais_courants_pct  NUMERIC,
    frais_entree_pct    NUMERIC,
    frais_sortie_pct    NUMERIC,
    asset_class     TEXT,
    investment_region TEXT,
    management_style   TEXT,
    objective_summary  TEXT,
    benchmark       TEXT,
    sfdr_classification TEXT,
    main_risks      TEXT,
    nav_frequency   TEXT,
    liquidity_constraints TEXT,
    performance_fee_pct NUMERIC,
    management_fees_pct   NUMERIC,
    transaction_costs_pct NUMERIC,
    other_costs_pct       NUMERIC,
    currency        TEXT,
    management_company TEXT,
    source_pdf      TEXT,
    quantity        NUMERIC,
    symbol          TEXT,
    archived_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_products_global_isin ON products_global(isin);
CREATE INDEX IF NOT EXISTS idx_products_global_fond ON products_global(fond);

-- Portefeuille propre à chaque utilisateur (favoris / suivis)
CREATE TABLE IF NOT EXISTS user_portfolio (
    user_id     TEXT NOT NULL,
    isin        TEXT NOT NULL REFERENCES products_global(isin) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY(user_id, isin)
);

-- Produits masqués par l’utilisateur (ne pas afficher dans sa vue)
CREATE TABLE IF NOT EXISTS user_hidden (
    user_id     TEXT NOT NULL,
    isin        TEXT NOT NULL REFERENCES products_global(isin) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY(user_id, isin)
);

-- Notes/étiquettes optionnelles par utilisateur
CREATE TABLE IF NOT EXISTS user_notes (
    user_id     TEXT NOT NULL,
    isin        TEXT NOT NULL REFERENCES products_global(isin) ON DELETE CASCADE,
    note        TEXT,
    updated_at  TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY(user_id, isin)
);
"""

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
MAX_OPENAI_CHARS = int(os.getenv("OPENAI_MAX_CHARS", "50000"))
ASSISTANT_MODEL = os.getenv("OPENAI_ASSISTANT_MODEL", "gpt-4o-mini")
ANALYST_REPORT_DIR = Path("data/analyste")
ASSISTANT_ID_FILE = ANALYST_REPORT_DIR / "assistant_id.txt"
ANALYST_MANAGER_URLS = {
    "AMUNDI": "https://funds.amundi.com/dl/doc/monthly-factsheet/{isin}/FRA/FRA/RETAIL/CRCA",
}


VIEW_SQL = """
SELECT
    g.*,
    (p.isin IS NOT NULL) AS in_portfolio
FROM products_global g
LEFT JOIN user_hidden h
    ON h.isin = g.isin AND h.user_id = %(user_id)s
LEFT JOIN user_portfolio p
    ON p.isin = g.isin AND p.user_id = %(user_id)s
WHERE g.archived_at IS NULL
  AND h.isin IS NULL
ORDER BY g.product_name NULLS LAST, g.isin
LIMIT %(limit)s;
"""


def get_db_url(arg_url: str) -> str:
    url = arg_url or os.getenv("DATABASE_URL") or ""
    if not url:
        raise SystemExit("DATABASE_URL manquant (ou --db-url).")
    return url


def print_rows(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("Aucune ligne.")
        return
    headers = ["fond", "product_name", "isin", "sri", "horizon", "frais_courants_pct", "management_company", "in_portfolio"]
    print(" | ".join(h.ljust(22) for h in headers))
    print("-" * 120)
    for r in rows:
        line = [
            str(r.get("fond", ""))[:20].ljust(20),
            str(r.get("product_name", ""))[:30].ljust(30),
            str(r.get("isin", ""))[:15].ljust(15),
            str(r.get("sri", ""))[:3].ljust(3),
            str(r.get("horizon", ""))[:15].ljust(15),
            str(r.get("frais_courants_pct", ""))[:8].ljust(8),
            str(r.get("management_company", ""))[:20].ljust(20),
            "★" if r.get("in_portfolio") else "",
        ]
        print(" | ".join(line))


def print_product_details(row: Dict[str, Any]) -> None:
    if not row:
        print("Aucun produit trouvé.")
        return
    fields = [
        ("Produit", row.get("product_name", "")),
        ("ISIN", row.get("isin", "")),
        ("Fond", row.get("fond", "")),
        ("SRI", row.get("sri", "")),
        ("Horizon", row.get("horizon", "")),
        ("Frais courants", row.get("frais_courants_pct", "")),
        ("Frais entrée", row.get("frais_entree_pct", "")),
        ("Frais sortie", row.get("frais_sortie_pct", "")),
        ("Classe d'actif", row.get("asset_class", "")),
        ("Région", row.get("investment_region", "")),
        ("Style", row.get("management_style", "")),
        ("Objectif", row.get("objective_summary", "")),
        ("Benchmark", row.get("benchmark", "")),
        ("SFDR", row.get("sfdr_classification", "")),
        ("Risques", row.get("main_risks", "")),
        ("Fréq. VL", row.get("nav_frequency", "")),
        ("Frais perf.", row.get("performance_fee_pct", "")),
        ("Frais gestion", row.get("management_fees_pct", "")),
        ("Frais transaction", row.get("transaction_costs_pct", "")),
        ("Autres frais", row.get("other_costs_pct", "")),
        ("Devise", row.get("currency", "")),
        ("Soc. gestion", row.get("management_company", "")),
        ("PDF", row.get("source_pdf", "")),
    ]
    for lbl, val in fields:
        print(f"{lbl}: {val}")


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


def display_with_curses(rows: List[Dict[str, Any]], db_url: str, user_id: Optional[str] = None) -> None:
    def _inner(stdscr):
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.curs_set(0)

        headers = ["Fond", "Produit", "ISIN", "SRI", "Horizon", "Frais", "Soc. gestion", "★", "Quantité", "Symbole"]
        col_widths = [12, 24, 15, 4, 18, 10, 18, 2, 10, 12]
        selected = 0
        top_offset = 0
        mode = "rows"  # rows | companies
        filter_company: Optional[str] = None
        filter_fond: Optional[str] = None
        sort_by_fees = False
        sort_by_risk = 0  # 0 none, 1 asc, -1 desc
        rows_list = list(rows)

        def draw_row(y: int, cols: List[str], highlight_risk: Any = None, is_selected: bool = False):
            x = 1
            color = curses.color_pair(_risk_color_pair(highlight_risk) if highlight_risk else 0)
            if is_selected:
                color |= curses.A_REVERSE
            for idx, col in enumerate(cols):
                col_str = "" if col is None else str(col)
                cell = col_str[: col_widths[idx]].ljust(col_widths[idx])
                stdscr.addstr(y, x, cell, color)
                x += col_widths[idx] + 1

        def show_popup(row: Dict[str, Any]) -> None:
            fields = [
                ("Produit", row.get("product_name", "")),
                ("ISIN", row.get("isin", "")),
                ("Fond", row.get("fond", "")),
                ("Quantité", row.get("quantity", "")),
                ("Symbole", row.get("symbol", "")),
                ("SRI", row.get("sri", "")),
                ("Horizon", row.get("horizon", "")),
                ("Frais courants", row.get("frais_courants_pct", "")),
                ("Frais entrée", row.get("frais_entree_pct", "")),
                ("Frais sortie", row.get("frais_sortie_pct", "")),
                ("Classe d'actif", row.get("asset_class", "")),
                ("Région", row.get("investment_region", "")),
                ("Style", row.get("management_style", "")),
                ("Objectif", row.get("objective_summary", "")),
                ("Benchmark", row.get("benchmark", "")),
                ("SFDR", row.get("sfdr_classification", "")),
                ("Risques", row.get("main_risks", "")),
                ("Fréq. VL", row.get("nav_frequency", "")),
                ("Frais perf.", row.get("performance_fee_pct", "")),
                ("Frais gestion", row.get("management_fees_pct", "")),
                ("Frais transaction", row.get("transaction_costs_pct", "")),
                ("Autres frais", row.get("other_costs_pct", "")),
                ("Soc. gestion", row.get("management_company", "")),
                ("Devise", row.get("currency", "")),
                ("PDF", row.get("source_pdf", "")),
            ]
            max_label = max(len(lbl) for lbl, _ in fields)
            width = min(curses.COLS - 2, 100)
            wrapped: list[str] = []
            pairs: list[tuple[str, str]] = []
            for lbl, val in fields:
                text = str(val or "")
                if len(text) > width - max_label - 6:
                    chunks = textwrap.wrap(text, width=width - max_label - 6, replace_whitespace=False)
                    if not chunks:
                        chunks = [""]
                    pairs.append((lbl, chunks[0]))
                    for extra in chunks[1:]:
                        pairs.append(("", extra))
                else:
                    pairs.append((lbl, text))
            h = min(curses.LINES - 2, max(10, len(pairs) + 4))
            start_y = max(1, (curses.LINES - h) // 2)
            start_x = max(1, (curses.COLS - width) // 2)
            win = curses.newwin(h, width, start_y, start_x)
            win.box()
            win.addstr(0, 2, " Détails produit ")
            y = 1
            for lbl, val in pairs:
                if y >= h - 2:
                    break
                line = f"{lbl.ljust(max_label)} : {val}"
                win.addstr(y, 2, line[: width - 4])
                y += 1
            win.addstr(h - 2, 2, "Enter/ESC/q: fermer | x: quantité | f: fond | s: symbole | t: tendance")
            win.refresh()
            while True:
                ch = win.getch()
                if ch in (ord("q"), ord("Q"), 27, ord("\n"), curses.KEY_ENTER, 10, 13):
                    break
                if ch in (ord("s"), ord("S")) and user_id:
                    sym = prompt_symbol()
                    if sym is not None:
                        try:
                            row["symbol"] = sym
                            upsert_product(db_url, row)
                            show_message("Succès", f"Symbole mis à jour : {sym}", color_pair=1)
                        except Exception as exc:
                            show_message("Erreur", f"Mise à jour symbole échouée:\n{exc}", color_pair=4)
                    render()
                    break
                if ch in (ord("f"), ord("F")) and user_id:
                    fonds = sorted({str(r.get("fond", "")) or "default" for r in rows_list})
                    fond_choice = prompt_fond_choice(fonds)
                    if fond_choice is not None:
                        try:
                            row["fond"] = fond_choice
                            upsert_product(db_url, row)
                            show_message("Succès", f"Fond mis à jour : {fond_choice}", color_pair=1)
                        except Exception as exc:
                            show_message("Erreur", f"Mise à jour fond échouée:\n{exc}", color_pair=4)
                    render()
                    break
                if ch in (ord("x"), ord("X")) and user_id:
                    qty = prompt_quantity()
                    if qty is not None:
                        try:
                            row["quantity"] = qty
                            row["source_pdf"] = row.get("source_pdf", "")
                            upsert_product(db_url, row)
                        except Exception as exc:
                            show_message("Erreur", f"Mise à jour quantité échouée:\n{exc}", color_pair=4)
                    render()
                    break
                if ch in (ord("t"), ord("T")):
                    symbol = (row.get("symbol") or "").strip()
                    if not symbol:
                        show_message("Tendance", "Aucun symbole renseigné pour ce produit.", color_pair=4)
                        render()
                        break

                    result_text = ""
                    stop_event = threading.Event()

                    def worker_tendance():
                        nonlocal result_text
                        try:
                            buf = io.StringIO()
                            with contextlib.redirect_stdout(buf):
                                tmt.build_medium_term_view(symbol)
                            result_text = buf.getvalue().strip() or "Aucun résultat."
                        except Exception as exc:
                            logging.exception("Erreur tendance moyen terme")
                            result_text = f"Erreur tendance: {exc}"
                        finally:
                            stop_event.set()

                    t_thread = threading.Thread(target=worker_tendance, daemon=True)
                    t_thread.start()
                    spin = ["|", "/", "-", "\\"]
                    idx_spin = 0
                    h_spin = 5
                    w_spin = 70
                    sy = max(1, (curses.LINES - h_spin) // 2)
                    sx = max(1, (curses.COLS - w_spin) // 2)
                    spin_win = curses.newwin(h_spin, w_spin, sy, sx)
                    spin_win.bkgd(" ", curses.color_pair(2))
                    spin_win.box()
                    while not stop_event.is_set():
                        spin_win.erase()
                        spin_win.bkgd(" ", curses.color_pair(2))
                        spin_win.box()
                        msg = f"Recherche cotation sur AlphaVantage... {spin[idx_spin % len(spin)]}"
                        spin_win.addstr(2, 2, msg[: w_spin - 4], curses.color_pair(2))
                        spin_win.refresh()
                        idx_spin += 1
                        time.sleep(0.1)
                    t_thread.join()
                    del spin_win
                    _show_report_popup(result_text)
                    render()
                    break
                if ch in (ord("0"),):
                    try:
                        # Si un rapport existe déjà, on l'affiche directement
                        cached = ANALYST_REPORT_DIR / f"rapport-{row.get('isin', '')}.txt"
                        if cached.exists():
                            report = cached.read_text()
                        else:
                            # Lancer un popup de progression uniquement si un reporting PDF est présent
                            pdf_exists = (ANALYST_REPORT_DIR / f"reporting-{row.get('isin', '')}.pdf").exists()
                            if pdf_exists:
                                report_text = "Analyse en cours..."
                                h2 = 5
                                w2 = 60
                                start_y = max(1, (curses.LINES - h2) // 2)
                                start_x = max(1, (curses.COLS - w2) // 2)
                                win2 = curses.newwin(h2, w2, start_y, start_x)
                                win2.bkgd(" ", curses.color_pair(2))
                                win2.box()
                                stop_event = threading.Event()

                                def worker():
                                    nonlocal report_text
                                    try:
                                        report_text = analyst_report(row.get("isin", ""), row.get("management_company", ""))
                                    except Exception as exc:
                                        report_text = f"Erreur lors de l'analyse: {exc}"
                                    finally:
                                        stop_event.set()

                                t = threading.Thread(target=worker, daemon=True)
                                t.start()
                                spinner = ["|", "/", "-", "\\"]
                                idx = 0
                                while not stop_event.is_set():
                                    win2.erase()
                                    win2.bkgd(" ", curses.color_pair(2))
                                    win2.box()
                                    msg = f"Analyse en cours par l'Assistant IA... {spinner[idx % len(spinner)]}"
                                    win2.addstr(2, 2, msg[: w2 - 4], curses.color_pair(2))
                                    win2.refresh()
                                    idx += 1
                                    time.sleep(0.1)
                                t.join()
                                # Afficher l'état final et attendre validation avant fermeture
                                win2.erase()
                                win2.bkgd(" ", curses.color_pair(2))
                                win2.box()
                                win2.addstr(2, 2, "Analyse terminée. Enter pour afficher le rapport.")
                                win2.refresh()
                                while True:
                                    ch2 = win2.getch()
                                    if ch2 in (10, 13, ord("\n"), ord("q"), ord("Q"), 27):
                                        break
                                del win2
                                report = report_text
                            else:
                                # Pas de PDF ni rapport : on passe au flux introuvable
                                report = "Reporting introuvable."
                    except Exception as exc:
                        report = f"Erreur lors de l'analyse: {exc}"
                    if report.startswith("Reporting introuvable"):
                        confirm = prompt_confirmation(
                            "Téléchargement manuel",
                            f"{report}\nAppuyez sur 'u' pour saisir une URL ou ESC pour annuler.",
                        )
                        if confirm:
                            url = prompt_url()
                            if url:
                                dest = ANALYST_REPORT_DIR / f"reporting-{row.get('isin', '')}.pdf"
                                try:
                                    info_win = curses.newwin(5, 60, max(1, (curses.LINES - 5) // 2), max(1, (curses.COLS - 60) // 2))
                                    info_win.bkgd(" ", curses.color_pair(1))
                                    info_win.box()
                                    info_win.addstr(2, 2, "Téléchargement en cours...")
                                    info_win.refresh()
                                    tmp = download_pdf(url, ANALYST_REPORT_DIR)
                                    del info_win
                                    tmp.rename(dest)
                                    # Spinner d'analyse
                                    h3 = 5
                                    w3 = 70
                                    s_y = max(1, (curses.LINES - h3) // 2)
                                    s_x = max(1, (curses.COLS - w3) // 2)
                                    spin_win = curses.newwin(h3, w3, s_y, s_x)
                                    spin_win.bkgd(" ", curses.color_pair(2))
                                    spin_win.box()
                                    stop_event = threading.Event()
                                    report_tmp = ""

                                    def worker2():
                                        nonlocal report_tmp
                                        try:
                                            report_tmp = analyst_report(row.get("isin", ""), row.get("management_company", ""))
                                        except Exception as exc:
                                            report_tmp = f"Erreur lors de l'analyse: {exc}"
                                        finally:
                                            stop_event.set()

                                    t2 = threading.Thread(target=worker2, daemon=True)
                                    t2.start()
                                    spinner = ["|", "/", "-", "\\"]
                                    idx2 = 0
                                    while not stop_event.is_set():
                                        spin_win.erase()
                                        spin_win.bkgd(" ", curses.color_pair(2))
                                        spin_win.box()
                                        msg = f"Analyse en cours par l'Assistant IA... {spinner[idx2 % len(spinner)]}"
                                        spin_win.addstr(2, 2, msg[: w3 - 4], curses.color_pair(2))
                                        spin_win.refresh()
                                        idx2 += 1
                                        time.sleep(0.1)
                                    t2.join()
                                    spin_win.erase()
                                    spin_win.bkgd(" ", curses.color_pair(2))
                                    spin_win.box()
                                    spin_win.addstr(2, 2, "Analyse terminée. Enter pour continuer.")
                                    spin_win.refresh()
                                    while True:
                                        ch3 = spin_win.getch()
                                        if ch3 in (10, 13, ord("\n"), ord("q"), ord("Q"), 27):
                                            break
                                    del spin_win
                                    report = report_tmp
                                except Exception as exc:
                                    try:
                                        del info_win
                                    except Exception:
                                        pass
                                    show_message("Erreur", f"Téléchargement échoué:\n{exc}", color_pair=4)
                                    report = f"Téléchargement échoué: {exc}"
                    _show_report_popup(report)
            del win

        def prompt_isin() -> Optional[str]:
            prompt_h = 5
            prompt_w = 40
            start_y = max(1, (curses.LINES - prompt_h) // 2)
            start_x = max(1, (curses.COLS - prompt_w) // 2)
            win = curses.newwin(prompt_h, prompt_w, start_y, start_x)
            win.box()
            win.addstr(0, 2, " Saisir ISIN ")
            win.addstr(2, 2, "ISIN: ")
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
                elif 32 <= ch <= 126 and len(buffer) < 20:
                    buffer += chr(ch)
                win.addstr(2, 7, " " * (prompt_w - 9))
                win.addstr(2, 7, buffer)
                win.move(2, min(prompt_w - 2, 7 + len(buffer)))
                win.refresh()
            curses.curs_set(0)
            del win
            return buffer.strip().upper() if buffer.strip() else None

        def prompt_fond_choice(fonds: List[str]) -> Optional[str]:
            if not fonds:
                return None
            fonds = ["All"] + fonds
            prompt_h = min(len(fonds) + 4, curses.LINES - 2)
            prompt_w = min(max(len(f) for f in fonds) + 20, curses.COLS - 2)
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
                hint = "Enter: valider / q/ESC: annuler"
                win.addstr(prompt_h - 2, 2, hint[: max(0, prompt_w - 4)])
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
            prompt_w = min(max(len(f) for f in options) + 14, curses.COLS - 2)
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
                hint = "Enter: valider / q/ESC: annuler"
                win.addstr(prompt_h - 2, 2, hint[: max(0, prompt_w - 4)])
                win.refresh()
                ch = win.getch()
                if ch in (curses.KEY_UP, ord("k")):
                    idx = max(0, idx - 1)
                elif ch in (curses.KEY_DOWN, ord("j")):
                    idx = min(len(options) - 1, idx + 1)
                elif ch in (ord("q"), ord("Q"), 27):
                    del win
                    return None
                elif ch in (ord("\n"), 10, 13):
                    choice = options[idx]
                    del win
                    return choice

        def prompt_confirmation(title: str, message: str) -> bool:
            lines = message.splitlines()
            max_len = max(len(line) for line in lines) if lines else 0
            h = min(curses.LINES - 2, max(5, len(lines) + 3))
            w = min(curses.COLS - 2, max(30, max_len + 4))
            start_y = max(1, (curses.LINES - h) // 2)
            start_x = max(1, (curses.COLS - w) // 2)
            win = curses.newwin(h, w, start_y, start_x)
            win.keypad(True)
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

        def prompt_fond_name() -> Optional[str]:
            prompt_h = 5
            prompt_w = min(60, curses.COLS - 2)
            start_y = max(1, (curses.LINES - prompt_h) // 2)
            start_x = max(1, (curses.COLS - prompt_w) // 2)
            win = curses.newwin(prompt_h, prompt_w, start_y, start_x)
            win.box()
            win.addstr(0, 2, " Nom du nouveau fond ")
            win.addstr(2, 2, "Nom: ")
            curses.curs_set(1)
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
                win.addstr(2, 7, buffer[: prompt_w - 9])
                win.move(2, min(prompt_w - 2, 7 + len(buffer[: prompt_w - 9])))
                win.refresh()
            curses.curs_set(0)
            del win
            return buffer.strip() if buffer.strip() else None

        def prompt_delete_confirmation(row: Dict[str, Any]) -> Optional[bool]:
            product = row.get("product_name", "") or "(sans nom)"
            isin_val = row.get("isin", "") or "(ISIN?)"
            msg_lines = [
                f"Supprimer '{product}'",
                f"ISIN: {isin_val}",
                "Supprimer aussi les fichiers PDF/rapports ?",
                "[o] Oui (avec fichiers) | [n] Oui (garder fichiers) | ESC annuler",
            ]
            h = min(curses.LINES - 2, max(7, len(msg_lines) + 3))
            w = min(curses.COLS - 2, max(40, max(len(m) for m in msg_lines) + 4))
            start_y = max(1, (curses.LINES - h) // 2)
            start_x = max(1, (curses.COLS - w) // 2)
            win = curses.newwin(h, w, start_y, start_x)
            curses.init_pair(7, curses.COLOR_YELLOW, curses.COLOR_RED)
            win.bkgd(" ", curses.color_pair(7))
            win.box()
            win.addstr(0, 2, " Confirmation suppression ")
            for i, line in enumerate(msg_lines[: h - 2]):
                win.addstr(1 + i, 2, line[: w - 4], curses.color_pair(7))
            win.refresh()
            while True:
                ch = win.getch()
                if ch in (ord("o"), ord("O")):
                    del win
                    return True
                if ch in (ord("n"), ord("N")):
                    del win
                    return False
                if ch in (27, ord("q"), ord("Q")):
                    del win
                    return None

        def prompt_path() -> Optional[str]:
            prompt_h = 5
            prompt_w = min(100, curses.COLS - 2)
            start_y = max(1, (curses.LINES - prompt_h) // 2)
            start_x = max(1, (curses.COLS - prompt_w) // 2)
            win = curses.newwin(prompt_h, prompt_w, start_y, start_x)
            win.box()
            win.addstr(0, 2, " Chemin du PDF ")
            win.addstr(2, 2, "Path: ")
            curses.curs_set(1)
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
                elif 32 <= ch <= 126 and len(buffer) < 200:
                    buffer += chr(ch)
                win.addstr(2, 8, " " * (prompt_w - 10))
                win.addstr(2, 8, buffer[: prompt_w - 10])
                win.move(2, min(prompt_w - 2, 8 + len(buffer[: prompt_w - 10])))
                win.refresh()
            curses.curs_set(0)
            del win
            return buffer.strip() if buffer.strip() else None

        def prompt_url() -> Optional[str]:
            prompt_h = 5
            prompt_w = min(120, curses.COLS - 2)
            start_y = max(1, (curses.LINES - prompt_h) // 2)
            start_x = max(1, (curses.COLS - prompt_w) // 2)
            win = curses.newwin(prompt_h, prompt_w, start_y, start_x)
            win.box()
            win.addstr(0, 2, " URL du PDF ")
            win.addstr(2, 2, "URL: ")
            curses.curs_set(1)
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
                elif 32 <= ch <= 126 and len(buffer) < 512:
                    buffer += chr(ch)
                win.addstr(2, 7, " " * (prompt_w - 9))
                win.addstr(2, 7, buffer[: prompt_w - 9])
                win.move(2, min(prompt_w - 2, 7 + len(buffer[: prompt_w - 9])))
                win.refresh()
            curses.curs_set(0)
            del win
            return buffer.strip() if buffer.strip() else None

        def prompt_quantity() -> Optional[float]:
            prompt_h = 5
            prompt_w = 40
            start_y = max(1, (curses.LINES - prompt_h) // 2)
            start_x = max(1, (curses.COLS - prompt_w) // 2)
            win = curses.newwin(prompt_h, prompt_w, start_y, start_x)
            win.box()
            win.addstr(0, 2, " Quantité ")
            win.addstr(2, 2, "Valeur: ")
            curses.curs_set(1)
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
                elif (chr(ch).isdigit() or chr(ch) in ".-,") and len(buffer) < 15:
                    buffer += chr(ch)
                win.addstr(2, 9, " " * (prompt_w - 11))
                win.addstr(2, 9, buffer[: prompt_w - 11])
                win.move(2, min(prompt_w - 2, 9 + len(buffer[: prompt_w - 11])))
                win.refresh()
            curses.curs_set(0)
            del win
            if not buffer.strip():
                return None
            try:
                return float(buffer.replace(",", "."))
            except Exception:
                return None

        def prompt_symbol() -> Optional[str]:
            prompt_h = 5
            prompt_w = 40
            start_y = max(1, (curses.LINES - prompt_h) // 2)
            start_x = max(1, (curses.COLS - prompt_w) // 2)
            win = curses.newwin(prompt_h, prompt_w, start_y, start_x)
            win.box()
            win.addstr(0, 2, " Symbole ")
            win.addstr(2, 2, "Ticker: ")
            curses.curs_set(1)
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
                elif 32 <= ch <= 126 and len(buffer) < 15:
                    buffer += chr(ch)
                win.addstr(2, 10, " " * (prompt_w - 12))
                win.addstr(2, 10, buffer[: prompt_w - 12])
                win.move(2, min(prompt_w - 2, 10 + len(buffer[: prompt_w - 12])))
                win.refresh()
            curses.curs_set(0)
            del win
            return buffer.strip() if buffer.strip() else None

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
                win.addstr(1 + i, 2, line[: w - 4], curses.color_pair(color_pair) if color_pair else 0)
            win.addstr(h - 2, 2, "Enter/ESC pour fermer")
            win.refresh()
            while True:
                ch = win.getch()
                if ch in (10, 13, 27, ord("q"), ord("Q")):
                    break
            del win

        def _clean_report_text(text: str) -> str:
            cleaned = text.replace("```", "")
            cleaned = cleaned.replace("**", "")
            cleaned = re.sub(r"\s+\|\s+", " | ", cleaned)
            cleaned = re.sub(r"[ \t]+", " ", cleaned)
            return cleaned.strip()

        def _show_report_popup(text: str, color_pair: int = 0):
            cleaned = _clean_report_text(text)
            paragraphs = cleaned.splitlines()
            wrapped: list[str] = []
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
                if ch == curses.KEY_UP:
                    scroll = max(0, scroll - 1)
                elif ch == curses.KEY_DOWN:
                    if scroll + (height - 3) < len(lines):
                        scroll += 1
                render_body()
            del win
            curses.flushinp()

        def add_product_flow():
            fonds_available = sorted({str(r.get("fond", "")) or "default" for r in rows_list})
            fond_choice = prompt_fond_choice_for_add(fonds_available)
            if not fond_choice:
                return
            if fond_choice == "<Nouveau>":
                fond_choice = prompt_fond_name()
                if not fond_choice:
                    return
            try:
                url = prompt_url()
                if not url:
                    return
                # Tous les PDF sont désormais stockés dans data/dic_pdfs (pas de sous-dossiers par fond)
                dest_dir = Path("data/dic_pdfs")
                dest_dir.mkdir(parents=True, exist_ok=True)
                try:
                    info_win = curses.newwin(5, 60, max(1, (curses.LINES - 5) // 2), max(1, (curses.COLS - 60) // 2))
                    info_win.bkgd(" ", curses.color_pair(1))
                    info_win.box()
                    info_win.addstr(2, 2, "Téléchargement en cours...")
                    info_win.refresh()
                    pdf_path = download_pdf(url, dest_dir)
                    del info_win
                    show_message("Téléchargement", f"Téléchargé: {pdf_path.name}", color_pair=1)
                except Exception as exc:
                    try:
                        del info_win
                    except Exception:
                        pass
                    show_message("Erreur", f"Téléchargement échoué:\n{exc}", color_pair=4)
                    return
                # Spinner pendant le parsing IA
                parse_text = "Parsing IA en cours..."
                h_parse = 5
                w_parse = 70
                sy = max(1, (curses.LINES - h_parse) // 2)
                sx = max(1, (curses.COLS - w_parse) // 2)
                parse_win = curses.newwin(h_parse, w_parse, sy, sx)
                parse_win.bkgd(" ", curses.color_pair(2))
                parse_win.box()
                stop_parse = threading.Event()
                payload: Dict[str, Any] = {}

                def parse_worker():
                    nonlocal payload
                    try:
                        payload = parse_pdf_to_payload(pdf_path, debug=False)
                    finally:
                        stop_parse.set()

                t_parse = threading.Thread(target=parse_worker, daemon=True)
                t_parse.start()
                spinner_parse = ["|", "/", "-", "\\"]
                idx_parse = 0
                while not stop_parse.is_set():
                    parse_win.erase()
                    parse_win.bkgd(" ", curses.color_pair(2))
                    parse_win.box()
                    msg = f"{parse_text} {spinner_parse[idx_parse % len(spinner_parse)]}"
                    parse_win.addstr(2, 2, msg[: w_parse - 4], curses.color_pair(2))
                    parse_win.refresh()
                    idx_parse += 1
                    time.sleep(0.1)
                t_parse.join()
                del parse_win
                isin_val = payload.get("isin")
                if not isin_val:
                    show_message("Erreur", "ISIN non détecté dans le PDF.", color_pair=4)
                    return
                if isin_val:
                    target_pdf = dest_dir / f"rapport-{isin_val}.pdf"
                    try:
                        if target_pdf.exists():
                            target_pdf.unlink()
                        pdf_path.rename(target_pdf)
                        pdf_path = target_pdf
                    except Exception:
                        pass
                payload["fond"] = fond_choice
                payload["source_pdf"] = str(pdf_path)
                upsert_product(db_url, payload)
                if user_id:
                    add_to_portfolio(db_url, user_id, payload.get("isin"))
                rows_list.append(payload)
                show_message("Succès", f"Produit ajouté : {payload.get('isin')}\nPDF: {Path(payload['source_pdf']).name}", color_pair=1)
            except Exception as exc:
                show_message("Erreur", f"Parsing/Upsert échoué:\n{exc}", color_pair=4)

        def render():
            stdscr.erase()
            stdscr.addstr(0, 1, "Pipeline Expert (DB)")
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
                    "Tab: Soc. gestion | 1: Fond | 2: +produit | 3: -produit | r: tri SRI | f: tri frais | d: ouvrir PDF | s: ISIN | Enter: détails | ESC/q: quitter",
                )
            else:
                current_rows = rows_list
                if filter_fond:
                    current_rows = [r for r in current_rows if str(r.get("fond", "")) == filter_fond]
                if filter_company:
                    current_rows = [r for r in current_rows if r.get("management_company", "") == filter_company]
                if sort_by_fees:
                    current_rows = sorted(
                        current_rows,
                        key=lambda r: float(r["frais_courants_pct"]) if str(r.get("frais_courants_pct", "")).replace(".", "", 1).isdigit() else float("inf"),
                    )
                if sort_by_risk != 0:
                    current_rows = sorted(
                        current_rows,
                        key=lambda r: int(r["sri"]) if str(r.get("sri", "")).isdigit() else (999 if sort_by_risk == 1 else -999),
                        reverse=True if sort_by_risk == -1 else False,
                    )
                draw_row(2, headers)
                y = 3
                visible_height = max(1, curses.LINES - 6)
                end_index = min(len(current_rows), top_offset + visible_height)
                for idx in range(top_offset, end_index):
                    row = current_rows[idx]
                    sri = str(row.get("sri", ""))
                    draw_row(
                        y,
                        [
                            row.get("fond", ""),
                            row.get("product_name", ""),
                            row.get("isin", ""),
                            sri,
                            row.get("horizon", ""),
                            row.get("frais_courants_pct", ""),
                            row.get("management_company", ""),
                            "★" if row.get("in_portfolio") else "",
                            row.get("quantity", ""),
                            row.get("symbol", ""),
                        ],
                        highlight_risk=sri,
                        is_selected=(idx == selected),
                    )
                    y += 1
                if end_index < len(current_rows):
                    stdscr.addstr(curses.LINES - 2, 1, "... (scroll down for more)")
                stdscr.addstr(
                    curses.LINES - 1,
                    1,
                    "Tab: Soc. gestion | 1: Fond | 2: +produit | 3: -produit | r: tri SRI | f: tri frais | d: ouvrir PDF | s: ISIN | Enter: détails | ESC/q: quitter",
                )
            stdscr.refresh()

        render()
        while True:
            ch = stdscr.getch()
            if ch in (ord("q"), ord("Q"), 27):
                break
            if ch == 9:  # Tab
                if mode == "rows":
                    mode = "companies"
                else:
                    mode = "rows"
                    filter_company = None
                selected = 0
                top_offset = 0
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
            if sort_by_fees:
                current_rows = sorted(
                    current_rows,
                    key=lambda r: float(r["frais_courants_pct"]) if str(r.get("frais_courants_pct", "")).replace(".", "", 1).isdigit() else float("inf"),
                )
            if sort_by_risk != 0:
                current_rows = sorted(
                    current_rows,
                    key=lambda r: int(r["sri"]) if str(r.get("sri", "")).isdigit() else (999 if sort_by_risk == 1 else -999),
                    reverse=True if sort_by_risk == -1 else False,
                )
            if ch in (curses.KEY_UP, ord("k")):
                selected = max(0, selected - 1)
                if selected < top_offset:
                    top_offset = selected
                render()
            elif ch in (curses.KEY_DOWN, ord("j")):
                selected = min(len(current_rows) - 1, selected + 1)
                visible_height = max(1, curses.LINES - 6)
                if selected >= top_offset + visible_height:
                    top_offset = max(0, selected - visible_height + 1)
                render()
            elif ch in (ord("\n"), curses.KEY_ENTER, 10, 13):
                if current_rows:
                    show_popup(current_rows[selected])
                    curses.flushinp()
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
                    if not pdf_path.exists():
                        show_message("Erreur", "Fichier introuvable.", color_pair=4)
                    elif pdf_path.suffix.lower() != ".pdf":
                        show_message("Erreur", f"Type non supporté ({pdf_path.suffix}). Ouvrez-le manuellement.", color_pair=4)
                    else:
                        try:
                            subprocess.Popen(["open", str(pdf_path.resolve())])
                        except Exception as exc:
                            show_message("Erreur", f"Ouverture impossible:\n{exc}", color_pair=4)
                render()
            elif ch in (ord("2"),):
                add_product_flow()
                selected = max(0, min(selected, len(rows_list) - 1))
                render()
            elif ch in (ord("1"),):
                fonds = sorted({str(r.get("fond", "")) or "default" for r in rows_list})
                choice = prompt_fond_choice(fonds)
                filter_fond = choice  # None clears the filter when "All" is chosen
                selected = 0
                top_offset = 0
                render()
            elif ch in (ord("3"),):
                if current_rows:
                    row = current_rows[selected]
                    choice = prompt_delete_confirmation(row)
                    if choice is not None:
                        rows_list = delete_product_db(db_url, rows_list, row, delete_files=choice)
                        selected = max(0, min(selected, len(rows_list) - 1))
                    render()

    curses.wrapper(_inner)


def delete_product_db(db_url: str, rows_list: List[Dict[str, Any]], row: Dict[str, Any], delete_files: bool) -> List[Dict[str, Any]]:
    isin_val = row.get("isin", "")
    source_pdf = row.get("source_pdf", "")
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM products_global WHERE isin=%(isin)s", {"isin": isin_val})
        conn.commit()
    if delete_files:
        try:
            if source_pdf:
                Path(source_pdf).unlink(missing_ok=True)
        except Exception:
            pass
        try:
            if isin_val:
                for suffix in ("rapport", "reporting"):
                    Path(f"data/analyste/{suffix}-{isin_val}.pdf").unlink(missing_ok=True)
                    Path(f"data/analyste/{suffix}-{isin_val}.txt").unlink(missing_ok=True)
        except Exception:
            pass
    return [
        r for r in rows_list
        if str(r.get("isin", "")).upper() != str(isin_val).upper()
    ]


# ---------------------
# Analyste (assistant OpenAI)
# ---------------------


def _ensure_reporting_pdf(isin: str, management_company: str) -> Optional[Path]:
    ANALYST_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    dest = ANALYST_REPORT_DIR / f"reporting-{isin}.pdf"
    if dest.exists():
        return dest
    url_template = None
    for key, url in ANALYST_MANAGER_URLS.items():
        if key.lower() in (management_company or "").lower():
            url_template = url
            break
    if not url_template:
        return None
    url = url_template.format(isin=isin)
    try:
        tmp = download_pdf(url, ANALYST_REPORT_DIR)
        tmp.rename(dest)
        return dest
    except Exception:
        return None


def _run_assistant_on_reporting(isin: str, management_company: str, debug: bool = False) -> str:
    pdf_path = ANALYST_REPORT_DIR / f"reporting-{isin}.pdf"
    if not pdf_path.exists():
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
    upload_id = None
    try:
        client = OpenAI()
        if ASSISTANT_ID_FILE.exists():
            assistant_id = ASSISTANT_ID_FILE.read_text().strip()
        else:
            instructions = """Tu es un assistant financier.

Contexte et objectif
- Tu reçois des reportings mensuels de fonds/portefeuilles au format PDF.
- Ta mission est d’extraire et de restituer, à partir du contenu du PDF, les informations de performance suivantes, puis de produire un résumé clair pour un client retail.

Données à extraire (obligatoire)
1) Performances glissantes du portefeuille
   - Extraire les performances glissantes (ex: 1 mois, 3 mois, 1 an, 3 ans, 5 ans, 10 ans, etc.).
   - Restituer en liste verticale (une ligne par horizon) et NON sous forme de tableau.
2) Performances calendaires par année
   - Extraire les performances par année (ex: 2024, 2023, 2022, ...).
   - Restituer en liste verticale (une ligne par année) et NON sous forme de tableau.

Exigences de format de sortie (obligatoire)
- La sortie doit être lisible dans un terminal Unix.
- Utiliser une mise en forme claire avec tabulations et alignement (indentation cohérente).
- Interdiction d’afficher ces données sous forme de table (pas de colonnes séparées type tableau). Uniquement des listes verticales.

Structure attendue de la réponse (obligatoire)
1) Performances glissantes
   - Titre: "### Performances glissantes"
   - Lignes au format:
     "\tDepuis le <horizon> : <valeur>%"
2) Performances calendaires
   - Titre: "### Performances calendaires"
   - Lignes au format:
     "\tAnnée <YYYY> : <valeur>%"
     (ou "\t<YYYY> : <valeur>%" si le PDF est présenté ainsi, mais rester cohérent au sein de la sortie)
3) Résumé client retail (clair, concis, pédagogique)
   - Expliquer ce que signifient ces performances (sans jargon inutile).
   - Mettre en évidence les points saillants: régularité, volatilité apparente via alternance d’années positives/négatives, tendance long terme vs court terme, etc.
4) Avis objectif basé sur les données
   - Émettre une appréciation factuelle: cohérence entre horizons glissants et années calendaires, présence de drawdowns/années négatives, qualité de la performance long terme.
   - Conclure si "cela ressemble à un bon fonds" ou "plutôt mitigé", en justifiant uniquement à partir des chiffres extraits (pas d’inventions).
   - Signaler explicitement toute donnée manquante ou ambiguë et éviter toute supposition non justifiée.

Règles de rigueur
- Ne pas inventer de valeurs. Si une donnée n’est pas trouvée dans le PDF, l’indiquer clairement.
- Respecter les formats de pourcentage tels qu’affichés (virgule décimale possible, signe négatif possible).
- Si plusieurs sections ressemblent aux performances, choisir celle correspondant au portefeuille/fonds principal du reporting (et préciser le choix si ambigu).

Exemple de rendu (indicatif, non contractuel)
### Performances glissantes
Depuis le 1 mois : 18,97%
Depuis le 3 mois : 0,28%
Depuis le 1 an : 6,30%
Depuis le 3 ans : 21,26%
Depuis le 5 ans : 55,93%
Depuis le 10 ans : 115,79%

### Performances calendaires
Année 2024 : 11,50%
Année 2023 : 22,75%
Année 2022 : -9,14%
Année 2021 : 23,86%
Année 2020 : -2,84%
Année 2019 : 28,78%
Année 2018 : -11,61%
"""
            assistant = client.beta.assistants.create(
                name="Analyse de reporting",
                instructions=instructions,
                model=ASSISTANT_MODEL,
                tools=[{"type": "file_search"}],
            )
            assistant_id = assistant.id
            ASSISTANT_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
            ASSISTANT_ID_FILE.write_text(assistant_id)
        with open(pdf_path, "rb") as f:
            upload = client.files.create(file=f, purpose="assistants")
            upload_id = upload.id
        thread = client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": f"Analyse le reporting PDF pour l'ISIN {isin}. Donne un résumé concis.",
                    "attachments": [{"file_id": upload_id, "tools": [{"type": "file_search"}]}],
                }
            ]
        )
        run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant_id)
        while run.status in ("queued", "in_progress", "requires_action"):
            time.sleep(0.5)
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        if run.status != "completed":
            return f"Échec analyse: statut {run.status}"
        messages = client.beta.threads.messages.list(thread_id=thread.id, order="desc", limit=1)
        for msg in messages.data:
            if msg.role == "assistant":
                text_parts = [c.text.value for c in msg.content if hasattr(c, "text")]
                result = "\n".join(text_parts).strip()
                cached_report.write_text(result)
                return result or "Réponse vide."
        return "Aucune réponse."
    except Exception as exc:
        return f"Erreur lors de l'analyse: {exc}"
    finally:
        if upload_id:
            try:
                client.files.delete(upload_id)
            except Exception:
                pass


def _analyst_report(isin: str, management_company: Optional[str] = None, debug: bool = False) -> str:
    if not isin:
        return "ISIN manquant."
    return analyst_report(isin, management_company or "", debug=debug)



# ---------------------
# Parsing PDF with OpenAI
# ---------------------


def extract_text_from_pdf(pdf_path: Path) -> str:
    parts: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            parts.append(txt)
    text = "\n".join(parts)
    if len(text) > MAX_OPENAI_CHARS:
        text = text[:MAX_OPENAI_CHARS]
    return text.strip()


def _prompt_openai(text: str) -> str:
    prompt = textwrap.dedent(
        """
        Tu es une IA spécialisée dans l'extraction d'informations de DIC/KID (fonds/UC).
        Fournis UNIQUEMENT un objet JSON avec exactement ces clés :
        {
          "isin": "...",
          "fond": "default",
          "product_name": "...",
          "sri": 3,
          "horizon": "...",
          "frais_courants_pct": 0.6,
          "frais_entree_pct": 0,
          "frais_sortie_pct": 0,
          "asset_class": "...",
          "investment_region": "...",
          "management_style": "...",
          "objective_summary": "...",
          "benchmark": null,
          "sfdr_classification": "...",
          "main_risks": "...",
          "nav_frequency": null,
          "liquidity_constraints": null,
          "performance_fee_pct": 0,
          "management_fees_pct": null,
          "transaction_costs_pct": null,
          "other_costs_pct": null,
          "currency": "...",
          "management_company": "...",
          "source_pdf": ""
        }
        Règles :
        - Ne rien ajouter d'autre (pas de texte hors JSON).
        - Les nombres doivent être des nombres JSON (pas de % ni de texte).
        - Mets "fond": "default" si non précisé.
        """
    )
    client = OpenAI()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content or ""


def clean_orphan_pdfs(db_url: str, base_dir: Path, force: bool = False) -> None:
    """Trouve les PDF dans base_dir non référencés en DB (champ source_pdf) et les affiche/efface."""
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT source_pdf FROM products_global WHERE archived_at IS NULL AND source_pdf IS NOT NULL")
            rows = cur.fetchall()
    referenced = set()
    for (path_str,) in rows:
        try:
            resolved = str(Path(path_str).resolve())
            referenced.add(resolved)
        except Exception:
            continue

    orphans = []
    for pdf in base_dir.rglob("*.pdf"):
        try:
            resolved = str(pdf.resolve())
        except Exception:
            continue
        if resolved not in referenced:
            orphans.append(pdf)

    if not orphans:
        print("Aucun PDF orphelin trouvé.")
        return

    action = "Suppression" if force else "Simulation"
    print(f"{action} : {len(orphans)} PDF orphelins détectés dans {base_dir}")
    for pdf in orphans:
        print(f" - {pdf}")
        if force:
            try:
                pdf.unlink()
            except Exception as exc:
                print(f"   (erreur suppression: {exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline Expert DB — schéma + lecture Postgres")
    parser.add_argument("--print-sql", action="store_true", help="Afficher le SQL de création")
    parser.add_argument("--write", help="Écrire le SQL dans le fichier spécifié")
    parser.add_argument("--db-url", help="URL Postgres (sinon DATABASE_URL)")
    parser.add_argument("--user", help="Identifiant utilisateur pour la vue combinée")
    parser.add_argument("--limit", type=int, default=20, help="Nombre de lignes à afficher (par défaut 20)")
    parser.add_argument("--upsert-json", help="Chemin d'un fichier JSON contenant les champs à upserter dans products_global (isin obligatoire)")
    parser.add_argument("--add-portfolio", action="store_true", help="(déprécié) --user suffit pour ajouter au portefeuille")
    parser.add_argument("--ui", action="store_true", help="Afficher la vue curses des produits (requiert --user)")
    parser.add_argument("--file", help="Parser un PDF DIC/KID et upserter le résultat en base")
    parser.add_argument("--debug", action="store_true", help="Afficher les réponses OpenAI brutes et logs détaillés")
    parser.add_argument("--show-isin", help="Afficher en texte les détails d'un produit par ISIN")
    parser.add_argument("--log-file", help="Chemin du fichier log (activé en mode --debug)")
    parser.add_argument("--clean", action="store_true", help="Lister les PDF orphelins dans data/ (non référencés en DB)")
    parser.add_argument("-f", "--force", dest="force_clean", action="store_true", help="Avec --clean, effacer réellement les PDF orphelins")
    args = parser.parse_args()

    # Configure logging (redacted formatter)
    _setup_logging(args.debug, args.log_file if args.debug else None)
    # Log uncaught exceptions
    def _log_excepthook(exc_type, exc_value, exc_traceback):
        logging.exception("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = _log_excepthook

    if args.write:
        path = Path(args.write)
        path.write_text(CREATE_TABLES_SQL.strip() + "\n", encoding="utf-8")
        print(f"Schéma écrit dans {path}")
        return
    if args.clean:
        db_url = get_db_url(args.db_url)
        clean_orphan_pdfs(db_url, Path("data"), force=args.force_clean)
        return

    if args.print_sql and not args.user and not args.upsert_json:
        sys.stdout.write(CREATE_TABLES_SQL.strip() + "\n")
        return

    if args.upsert_json:
        db_url = get_db_url(args.db_url)
        payload = json.loads(Path(args.upsert_json).read_text())
        if not payload.get("isin"):
            raise SystemExit("Upsert JSON invalide : champ 'isin' manquant.")
        upsert_product(db_url, payload)
        if args.user:
            add_to_portfolio(db_url, args.user, payload.get("isin"))
        print("Upsert terminé.")
        return

    if args.file:
        db_url = get_db_url(args.db_url)
        pdf_path = Path(args.file)
        payload = parse_pdf_to_payload(pdf_path, debug=args.debug)
        if not payload.get("isin"):
            raise SystemExit(f"Parsing impossible : ISIN manquant dans {pdf_path}")
        upsert_product(db_url, payload)
        if args.user:
            add_to_portfolio(db_url, args.user, payload.get("isin"))
        print(f"PDF parsé et upserté : {pdf_path}")
        return

    if args.show_isin:
        db_url = get_db_url(args.db_url)
        row = fetch_one_by_isin(db_url, args.show_isin)
        print_product_details(row)
        return

    if args.user and args.ui:
        db_url = get_db_url(args.db_url)
        rows = fetch_rows(db_url, args.user, args.limit)
        try:
            display_with_curses(rows, db_url, user_id=args.user)
        except Exception:
            logging.exception("Erreur lors de l'exécution de l'UI")
            raise
        return

    if args.user:
        db_url = get_db_url(args.db_url)
        rows = fetch_rows(db_url, args.user, args.limit)
        print_rows(rows)
        return

    # Par défaut, afficher le SQL si aucune autre option n'est fournie
    sys.stdout.write(CREATE_TABLES_SQL.strip() + "\n")


if __name__ == "__main__":
    main()
