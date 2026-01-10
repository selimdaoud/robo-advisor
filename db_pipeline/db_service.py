import psycopg
from typing import Any, Dict, List, Optional
import logging

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


def fetch_rows(db_url: str, user_id: str, limit: int) -> List[Dict[str, Any]]:
    with psycopg.connect(db_url) as conn:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(VIEW_SQL, {"user_id": user_id, "limit": limit})
            return list(cur.fetchall())


def fetch_one_by_isin(db_url: str, isin: str) -> Optional[Dict[str, Any]]:
    sql = """
    SELECT *
    FROM products_global
    WHERE isin = %(isin)s
      AND archived_at IS NULL
    """
    with psycopg.connect(db_url) as conn:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(sql, {"isin": isin})
            row = cur.fetchone()
            return dict(row) if row else None


def upsert_product(db_url: str, payload: Dict[str, Any]) -> None:
    cols_allowed = {
        "isin",
        "fond",
        "product_name",
        "sri",
        "horizon",
        "frais_courants_pct",
        "frais_entree_pct",
        "frais_sortie_pct",
        "quantity",
        "symbol",
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
    }
    data = {
        k: v
        for k, v in payload.items()
        if k in cols_allowed and not (v is None or (isinstance(v, str) and v.strip() == ""))
    }
    if "isin" not in data or not data["isin"]:
        raise ValueError("Le champ 'isin' est obligatoire pour l'upsert.")
    columns = list(data.keys())
    placeholders = ", ".join([f"%({c})s" for c in columns])
    sets = ", ".join([f"{c}=EXCLUDED.{c}" for c in columns if c != "isin"])
    sql = f"""
    INSERT INTO products_global ({', '.join(columns)})
    VALUES ({placeholders})
    ON CONFLICT (isin) DO UPDATE
    SET {sets},
        updated_at = now()
    """
    logging.getLogger(__name__).debug("Upsert product data=%s", data)
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, data)
        conn.commit()


def add_to_portfolio(db_url: str, user_id: str, isin: str) -> None:
    sql = """
    INSERT INTO user_portfolio (user_id, isin)
    VALUES (%(user_id)s, %(isin)s)
    ON CONFLICT DO NOTHING
    """
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"user_id": user_id, "isin": isin})
        conn.commit()
