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
