import json
import logging
import os
import re
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pdfplumber
from openai import OpenAI

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
ASSISTANT_MODEL = os.getenv("OPENAI_ASSISTANT_MODEL", "gpt-4o-mini")
MAX_OPENAI_CHARS = int(os.getenv("OPENAI_MAX_CHARS", "50000"))


def extract_text_from_pdf(pdf_path: Path) -> str:
    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    text = "\n".join(parts)
    if len(text) > MAX_OPENAI_CHARS:
        text = text[:MAX_OPENAI_CHARS]
        logging.info("Texte tronqué pour %s (len=%d > %d)", pdf_path.name, len(text), MAX_OPENAI_CHARS)
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
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": text}],
        temperature=0,
    )
    return resp.choices[0].message.content or ""


def _parse_openai_json(content: str, pdf_path: Path) -> Dict[str, Any]:
    cleaned = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            data.setdefault("fond", "default")
            data["source_pdf"] = str(pdf_path)
            return data
    except Exception as exc:
        logging.error("Échec parsing JSON pour %s: %s", pdf_path, exc)
    return {"isin": "", "fond": "default", "product_name": "", "source_pdf": str(pdf_path)}


def parse_pdf_to_payload(pdf_path: Path, debug: bool = False) -> Dict[str, Any]:
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)
    text = extract_text_from_pdf(pdf_path)
    if not text:
        raise ValueError(f"Texte vide pour {pdf_path}")
    content = _prompt_openai(text)
    if debug:
        logging.debug("OpenAI response for %s:\n%s", pdf_path, content)
        print(f"[DEBUG] OpenAI raw for {pdf_path}:\n{content}\n")
    return _parse_openai_json(content, pdf_path)


# Analyst assistant
ANALYST_REPORT_DIR = Path("data/analyste")
ASSISTANT_ID_FILE = ANALYST_REPORT_DIR / "assistant_id.txt"
ANALYST_MANAGER_URLS = {
    "AMUNDI": "https://funds.amundi.com/dl/doc/monthly-factsheet/{isin}/FRA/FRA/RETAIL/CRCA",
}


def _run_assistant_on_reporting(isin: str, management_company: str, pdf_path: Path, debug: bool = False) -> str:
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
            instructions = (
                "Tu es un assistant financier. "
                "À partir de reportings mensuels au format PDF, tu dois extraire : "
                "1) les performances glissantes du portefeuille, affiché verticalement, pas en mode tableau, "
                "2) les performances calendaires par année, affiché verticalement, pas en mode tableau, "
                "3) un résumé clair pour un client retail. tu dois émettre des avis objectifs basés sur les données "
                "et décrire si cela ressemble à un bon fonds ou pas. "
                "you must format clearly the output to be visible on a unix terminal with proper tabulation"
            )
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


def analyst_report(isin: str, management_company: Optional[str] = None, debug: bool = False) -> str:
    if not isin:
        return "ISIN manquant."
    pdf_path = ANALYST_REPORT_DIR / f"reporting-{isin}.pdf"
    if not pdf_path.exists():
        return (
            f"Reporting introuvable. Merci de télécharger manuellement le dernier rapport pour l'ISIN {isin} "
            f"et de le sauvegarder sous {pdf_path}"
        )
    return _run_assistant_on_reporting(isin, management_company or "", pdf_path, debug=debug)
