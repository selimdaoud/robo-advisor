"""
Simple connectivity test to Postgres.

Usage:
    export DATABASE_URL=postgresql://user:pass@host:5432/dbname
    python db_pipeline/test_psql.py

Optional args:
    python db_pipeline/test_psql.py --db-url postgresql://user:pass@host/db
"""

import argparse
import os

import psycopg


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Postgres connection")
    parser.add_argument("--db-url", help="Postgres URL (defaults to env DATABASE_URL)")
    args = parser.parse_args()

    db_url = args.db_url or os.getenv("DATABASE_URL")
    if not db_url:
        raise SystemExit("DATABASE_URL not set and --db-url not provided.")

    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()[0]
        print(f"Connection OK. Server version: {version}")
    except Exception as exc:
        print(f"Connection failed: {exc}")


if __name__ == "__main__":
    main()
