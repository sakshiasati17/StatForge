-- StatForge PostgreSQL Schema

CREATE TABLE IF NOT EXISTS experiments (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(255) NOT NULL UNIQUE,
    description     TEXT,
    status          VARCHAR(20) DEFAULT 'draft' CHECK (status IN ('draft', 'running', 'stopped', 'completed')),
    alpha           FLOAT NOT NULL DEFAULT 0.05,
    power_target    FLOAT NOT NULL DEFAULT 0.80,
    intended_split  FLOAT NOT NULL DEFAULT 0.50,
    start_date      TIMESTAMP,
    end_date        TIMESTAMP,
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS experiment_assignments (
    id              BIGSERIAL PRIMARY KEY,
    experiment_id   INT REFERENCES experiments(id) ON DELETE CASCADE,
    user_id         BIGINT NOT NULL,
    variant         VARCHAR(50) NOT NULL,
    assigned_at     TIMESTAMP DEFAULT NOW(),
    UNIQUE (experiment_id, user_id)
);

CREATE TABLE IF NOT EXISTS experiment_observations (
    id              BIGSERIAL PRIMARY KEY,
    experiment_id   INT REFERENCES experiments(id) ON DELETE CASCADE,
    user_id         BIGINT NOT NULL,
    metric_name     VARCHAR(100) NOT NULL,
    metric_value    FLOAT NOT NULL,
    pre_value       FLOAT,
    observed_at     TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS experiment_results (
    id                  SERIAL PRIMARY KEY,
    experiment_id       INT REFERENCES experiments(id) ON DELETE CASCADE,
    metric_name         VARCHAR(100) NOT NULL,
    test_type           VARCHAR(50),
    p_value             FLOAT,
    adjusted_p_value    FLOAT,
    effect_size         FLOAT,
    ci_lower            FLOAT,
    ci_upper            FLOAT,
    n_control           INT,
    n_treatment         INT,
    significant         BOOLEAN,
    srm_detected        BOOLEAN DEFAULT FALSE,
    variance_reduction  FLOAT,
    computed_at         TIMESTAMP DEFAULT NOW(),
    UNIQUE (experiment_id, metric_name)
);

CREATE INDEX idx_assignments_experiment ON experiment_assignments(experiment_id);
CREATE INDEX idx_assignments_user ON experiment_assignments(user_id);
CREATE INDEX idx_observations_experiment ON experiment_observations(experiment_id);
CREATE INDEX idx_observations_metric ON experiment_observations(metric_name);
CREATE INDEX idx_results_experiment ON experiment_results(experiment_id);
