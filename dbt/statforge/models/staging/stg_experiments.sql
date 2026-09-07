-- Staging: clean and type-cast raw experiments table
with source as (
    select * from {{ source('statforge', 'experiments') }}
),

cleaned as (
    select
        experiment_id,
        trim(experiment_name)                          as experiment_name,
        trim(description)                              as description,
        lower(trim(status))                            as status,
        cast(alpha as numeric(5,4))                    as alpha,
        cast(power as numeric(5,4))                    as target_power,
        cast(min_detectable_effect as numeric(10,6))   as min_detectable_effect,
        created_at::timestamptz                        as created_at,
        updated_at::timestamptz                        as updated_at
    from source
    where experiment_id is not null
)

select * from cleaned
