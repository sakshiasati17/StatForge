-- Mart: daily metric trends per experiment and group — feeds time-series charts
with obs as (
    select * from {{ ref('stg_observations') }}
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
        -- 7-day rolling average
        avg(avg(metric_value)) over (
            partition by experiment_id, group_name
            order by observed_date
            rows between 6 preceding and current row
        )                                                         as rolling_7d_mean,
        -- cumulative users
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
where group_name = 'treatment'  -- lift is treatment-relative; one row per day
