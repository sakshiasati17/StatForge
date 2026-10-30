-- Staging: clean raw experiment_observations joined with experiment_assignments
with obs as (
    select * from {{ source('statforge', 'experiment_observations') }}
),

assign as (
    select * from {{ source('statforge', 'experiment_assignments') }}
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

select * from cleaned
