-- Staging: clean raw experiment_observations
with source as (
    select * from {{ source('statforge', 'experiment_observations') }}
),

cleaned as (
    select
        observation_id,
        experiment_id,
        user_id,
        lower(trim(group_name))          as group_name,
        cast(metric_value as numeric)    as metric_value,
        cast(covariate_value as numeric) as covariate_value,
        observed_at::timestamptz         as observed_at,
        date_trunc('day', observed_at)   as observed_date
    from source
    where observation_id is not null
      and group_name in ('control', 'treatment')
)

select * from cleaned
