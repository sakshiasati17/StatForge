-- Staging: clean raw experiment_results
with source as (
    select * from {{ source('statforge', 'experiment_results') }}
),

cleaned as (
    select
        result_id,
        experiment_id,
        trim(metric_name)                        as metric_name,
        trim(test_type)                          as test_type,
        cast(p_value as numeric(10,6))           as p_value,
        cast(effect_size as numeric(10,6))       as effect_size,
        cast(ci_lower as numeric(12,6))          as ci_lower,
        cast(ci_upper as numeric(12,6))          as ci_upper,
        cast(n_control as integer)               as n_control,
        cast(n_treatment as integer)             as n_treatment,
        significant::boolean                     as significant,
        created_at::timestamptz                  as created_at
    from source
    where result_id is not null
      and p_value between 0 and 1
)

select * from cleaned
