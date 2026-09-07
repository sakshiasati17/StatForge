-- Custom test: n_control and n_treatment must be positive
select experiment_id, n_control, n_treatment
from {{ ref('mart_experiment_summary') }}
where n_control <= 0 or n_treatment <= 0
