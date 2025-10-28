act='gelu'
lag=2
num_states=15

echo Training sleep data on $lag lags and $num_states states
python train_neuroscience.py --exp_name sleep --activation $act --restarts_num 10 --num_states $num_states --device cuda:0 --lag $lag > logs_neuro/log_neuro_sleep_lag_$lag._num_states_$num_states._train_${act}.log

echo Training awake data on $lag lags and $num_states states
python train_neuroscience.py --exp_name awake --activation $act --restarts_num 10 --num_states $num_states --device cuda:0 --lag $lag > logs_neuro/log_neuro_awake_lag_$lag._num_states_$num_states._train_${act}.log


