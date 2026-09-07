-- Custom test: all p-values in mart must be between 0 and 1
select experiment_id, p_value
from {{ ref('mart_experiment_summary') }}
where p_value is not null
  and (p_value < 0 or p_value > 1)
