-- Mart: one row per experiment with key stats — feeds Tableau/BI dashboards
with experiments as (
    select * from {{ ref('stg_experiments') }}
),

results as (
    select * from {{ ref('stg_experiment_results') }}
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
    from {{ ref('stg_observations') }}
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
    -- one result row per experiment (pick primary metric = smallest p_value)
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
        e.min_detectable_effect,
        e.created_at,

        c.n_control,
        t.n_treatment,
        c.n_control + t.n_treatment                              as total_users,
        c.first_observation,
        c.last_observation,
        c.n_days_running,

        c.control_mean,
        t.treatment_mean,
        t.treatment_mean - c.control_mean                        as absolute_lift,
        case when c.control_mean = 0 then null
             else (t.treatment_mean - c.control_mean) / c.control_mean
        end                                                      as relative_lift_pct,

        r.metric_name,
        r.p_value,
        r.effect_size,
        r.ci_lower,
        r.ci_upper,
        r.significant,

        -- sample ratio mismatch flag (heuristic: >5% deviation from expected)
        case
            when abs(c.n_control::float / (c.n_control + t.n_treatment) - 0.5) > 0.05
            then true else false
        end                                                      as srm_flag

    from experiments e
    left join control_obs   c using (experiment_id)
    left join treatment_obs t using (experiment_id)
    left join primary_results r using (experiment_id)
)

select * from final
