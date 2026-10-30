"""
StatForge dbt Models & Analytics Builder.

Executes and validates the 5 dbt models against PostgreSQL:
- staging.stg_experiments (view)
- staging.stg_experiment_results (view)
- staging.stg_observations (view)
- marts.mart_experiment_summary (table)
- marts.mart_daily_metrics (table)

Runs test assertions:
- assert_no_negative_users
- assert_p_value_in_range
"""
import time
import psycopg2
from pathlib import Path


def build_models(
    dbname: str = "statforge",
    user: str = "sakshiasati",
    host: str = "localhost",
    port: int = 5432,
    password: str = "",
) -> None:
    print(f"Connecting to PostgreSQL ({user}@{host}:{port}/{dbname})...")
    conn = psycopg2.connect(
        dbname=dbname, user=user, host=host, port=port, password=password
    )
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("CREATE SCHEMA IF NOT EXISTS analytics;")
    print("Schema 'analytics' verified.\n")

    models = [
        (
            "staging.stg_experiments",
            "view",
            """
            CREATE OR REPLACE VIEW analytics.stg_experiments AS
            with source as (
                select * from public.experiments
            ),
            cleaned as (
                select
                    id                                             as experiment_id,
                    trim(name)                                     as experiment_name,
                    trim(description)                              as description,
                    lower(trim(status))                            as status,
                    cast(alpha as numeric(5,4))                    as alpha,
                    cast(power_target as numeric(5,4))             as target_power,
                    cast(intended_split as numeric(5,4))           as intended_split,
                    start_date::timestamptz                        as start_date,
                    end_date::timestamptz                          as end_date,
                    created_at::timestamptz                        as created_at,
                    updated_at::timestamptz                        as updated_at
                from source
                where id is not null
            )
            select * from cleaned;
            """,
        ),
        (
            "staging.stg_experiment_results",
            "view",
            """
            CREATE OR REPLACE VIEW analytics.stg_experiment_results AS
            with source as (
                select * from public.experiment_results
            ),
            cleaned as (
                select
                    id                                       as result_id,
                    experiment_id,
                    trim(metric_name)                        as metric_name,
                    trim(test_type)                          as test_type,
                    cast(p_value as numeric(10,6))           as p_value,
                    cast(adjusted_p_value as numeric(10,6))  as adjusted_p_value,
                    cast(effect_size as numeric(10,6))       as effect_size,
                    cast(ci_lower as numeric(12,6))          as ci_lower,
                    cast(ci_upper as numeric(12,6))          as ci_upper,
                    cast(n_control as integer)               as n_control,
                    cast(n_treatment as integer)             as n_treatment,
                    significant::boolean                     as significant,
                    srm_detected::boolean                    as srm_detected,
                    cast(variance_reduction as numeric(10,6)) as variance_reduction,
                    computed_at::timestamptz                 as created_at
                from source
                where id is not null
                  and p_value between 0 and 1
            )
            select * from cleaned;
            """,
        ),
        (
            "staging.stg_observations",
            "view",
            """
            CREATE OR REPLACE VIEW analytics.stg_observations AS
            with obs as (
                select * from public.experiment_observations
            ),
            assign as (
                select * from public.experiment_assignments
            ),
            cleaned as (
                select
                    obs.id                                        as observation_id,
                    obs.experiment_id,
                    obs.user_id,
                    lower(trim(assign.variant))                   as group_name,
                    trim(obs.metric_name)                         as metric_name,
                    cast(obs.metric_value as numeric)             as metric_value,
                    cast(obs.pre_value as numeric)                as covariate_value,
                    obs.observed_at::timestamptz                  as observed_at,
                    date_trunc('day', obs.observed_at)            as observed_date
                from obs
                inner join assign
                    on obs.experiment_id = assign.experiment_id
                   and obs.user_id = assign.user_id
                where obs.id is not null
                  and lower(trim(assign.variant)) in ('control', 'treatment')
            )
            select * from cleaned;
            """,
        ),
        (
            "marts.mart_experiment_summary",
            "table",
            """
            DROP TABLE IF EXISTS analytics.mart_experiment_summary CASCADE;
            CREATE TABLE analytics.mart_experiment_summary AS
            with experiments as (
                select * from analytics.stg_experiments
            ),
            results as (
                select * from analytics.stg_experiment_results
            ),
            obs_agg as (
                select
                    experiment_id,
                    group_name,
                    count(*)                        as n_users,
                    avg(metric_value)               as mean_metric,
                    stddev(metric_value)            as std_metric,
                    min(observed_date)              as first_observation,
                    max(observed_date)              as last_observation,
                    count(distinct observed_date)   as n_days_running
                from analytics.stg_observations
                group by experiment_id, group_name
            ),
            control_obs as (
                select experiment_id, n_users as n_control, mean_metric as control_mean,
                       std_metric as control_std, first_observation, last_observation, n_days_running
                from obs_agg where group_name = 'control'
            ),
            treatment_obs as (
                select experiment_id, n_users as n_treatment, mean_metric as treatment_mean,
                       std_metric as treatment_std
                from obs_agg where group_name = 'treatment'
            ),
            primary_results as (
                select distinct on (experiment_id)
                    experiment_id, metric_name, p_value, effect_size,
                    ci_lower, ci_upper, significant
                from results
                order by experiment_id, p_value asc
            ),
            final as (
                select
                    e.experiment_id,
                    e.experiment_name,
                    e.status,
                    e.alpha,
                    e.target_power,
                    e.intended_split,
                    e.created_at,

                    c.n_control,
                    t.n_treatment,
                    c.n_control + t.n_treatment                              as total_users,
                    c.first_observation,
                    c.last_observation,
                    c.n_days_running,

                    c.control_mean,
                    c.control_std,
                    t.treatment_mean,
                    t.treatment_std,

                    case
                        when c.control_mean = 0 then null
                        else (t.treatment_mean - c.control_mean) / c.control_mean
                    end                                                      as observed_lift_pct,

                    r.metric_name                                            as primary_metric,
                    r.p_value,
                    r.effect_size,
                    r.ci_lower,
                    r.ci_upper,
                    r.significant,

                    case
                        when abs(t.n_treatment::float / (c.n_control + t.n_treatment) - coalesce(e.intended_split, 0.5)) > 0.05
                        then true else false
                    end                                                      as srm_flag

                from experiments e
                left join control_obs   c on e.experiment_id = c.experiment_id
                left join treatment_obs t on e.experiment_id = t.experiment_id
                left join primary_results r on e.experiment_id = r.experiment_id
            )
            select * from final;
            """,
        ),
        (
            "marts.mart_daily_metrics",
            "table",
            """
            DROP TABLE IF EXISTS analytics.mart_daily_metrics CASCADE;
            CREATE TABLE analytics.mart_daily_metrics AS
            with obs as (
                select * from analytics.stg_observations
            ),
            daily as (
                select
                    experiment_id,
                    group_name,
                    observed_date,
                    count(*)                                                  as n_users,
                    avg(metric_value)                                         as daily_mean,
                    stddev(metric_value)                                      as daily_std,
                    sum(metric_value)                                         as daily_total,
                    avg(avg(metric_value)) over (
                        partition by experiment_id, group_name
                        order by observed_date
                        rows between 6 preceding and current row
                    )                                                         as rolling_7d_mean,
                    sum(count(*)) over (
                        partition by experiment_id, group_name
                        order by observed_date
                        rows unbounded preceding
                    )                                                         as cumulative_users
                from obs
                group by experiment_id, group_name, observed_date
            ),
            with_lift as (
                select
                    d.*,
                    ctrl.daily_mean                                           as control_daily_mean,
                    case when ctrl.daily_mean = 0 then null
                         else (d.daily_mean - ctrl.daily_mean) / ctrl.daily_mean
                    end                                                       as daily_lift_pct
                from daily d
                left join daily ctrl
                    on  ctrl.experiment_id = d.experiment_id
                    and ctrl.observed_date = d.observed_date
                    and ctrl.group_name    = 'control'
            )
            select * from with_lift
            where group_name = 'treatment';
            """,
        ),
    ]

    for model_name, mat_type, sql in models:
        t0 = time.time()
        print(f"Building {model_name} ({mat_type})...", end=" ", flush=True)
        cur.execute(sql)
        dt = time.time() - t0
        print(f"OK ({dt:.3f}s)")

    # Run tests
    print("\nRunning schema & business tests:")
    tests = [
        (
            "assert_no_negative_users",
            "SELECT count(*) FROM analytics.mart_experiment_summary WHERE total_users < 0 OR n_control < 0 OR n_treatment < 0;",
        ),
        (
            "assert_p_value_in_range",
            "SELECT count(*) FROM analytics.stg_experiment_results WHERE p_value < 0 OR p_value > 1;",
        ),
    ]

    for test_name, query in tests:
        cur.execute(query)
        failed_count = cur.fetchone()[0]
        if failed_count == 0:
            print(f"  PASS: {test_name}")
        else:
            raise ValueError(f"  FAIL: {test_name} returned {failed_count} violations!")

    cur.close()
    conn.close()
    print("\nAll 5 dbt models built and verified successfully.")


if __name__ == "__main__":
    build_models()
