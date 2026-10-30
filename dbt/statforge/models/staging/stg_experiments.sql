-- Staging: clean and type-cast raw experiments table
with source as (
    select * from {{ source('statforge', 'experiments') }}
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

select * from cleaned
