-- PostgreSQL schema + sample data for DSS

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(320) NOT NULL UNIQUE,
    password_hash VARCHAR(512) NOT NULL,
    role VARCHAR(32) NOT NULL DEFAULT 'user'
);
CREATE INDEX IF NOT EXISTS ix_users_email ON users (email);

CREATE TABLE IF NOT EXISTS recommendation_history (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    car_count INTEGER NOT NULL,
    summary VARCHAR(400) NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_recommendation_history_user_id ON recommendation_history (user_id);

CREATE TABLE IF NOT EXISTS saved_cars (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    title VARCHAR(200) NOT NULL DEFAULT '',
    source VARCHAR(32) NOT NULL DEFAULT 'manual',
    car_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_saved_cars_user_id ON saved_cars (user_id);

CREATE TABLE IF NOT EXISTS criteria_config (
    id SERIAL PRIMARY KEY,
    key VARCHAR(64) NOT NULL UNIQUE,
    label VARCHAR(200) NOT NULL,
    direction VARCHAR(16) NOT NULL,
    default_weight DOUBLE PRECISION NOT NULL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS ix_criteria_config_key ON criteria_config (key);

INSERT INTO criteria_config (key, label, direction, default_weight)
VALUES
    ('price', 'Gia xe', 'cost', 0.15),
    ('mileage', 'So km da di', 'cost', 0.10),
    ('year', 'Doi xe', 'benefit', 0.10),
    ('mpg', 'Tiet kiem nhien lieu', 'benefit', 0.10),
    ('accident_risk', 'Rui ro tai nan', 'cost', 0.15),
    ('maintenance_cost', 'Chi phi bao duong', 'cost', 0.15),
    ('seller_rating', 'Danh gia nguoi ban', 'benefit', 0.10),
    ('driver_rating', 'Danh gia nguoi dung', 'benefit', 0.10),
    ('one_owner', 'Mot chu su dung', 'benefit', 0.05)
ON CONFLICT (key) DO NOTHING;

INSERT INTO users (email, password_hash, role)
VALUES
    ('admin@example.com', 'admin123', 'admin'),
    ('user@example.com', '123456', 'user')
ON CONFLICT (email) DO UPDATE
SET
    password_hash = EXCLUDED.password_hash,
    role = EXCLUDED.role;
