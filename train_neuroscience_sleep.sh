
act='gelu'

for lag in 1 2 5 10 20
do
    for num_states in 1 3 5 10 15 20
    do
        echo Training on $lag lags and $num_states states
        python train_neuroscience.py --exp_name sleep --activation $act --restarts_num 20 --num_states $num_states --device cuda:2 --lag $lag > logs_neuro/log_neuro_sleep_lag_$lag._num_states_$num_states._train_${act}.log
    done
done



act='cos'

for lag in 1 2 5 10 20
do
    for num_states in 15 20 1 3 5 10
    do
        echo Training on $lag lags and $num_states states
        python train_neuroscience.py --exp_name sleep --activation $act --restarts_num 20 --num_states $num_states --device cuda:3 --lag $lag > logs_neuro/log_neuro_sleep_lag_$lag._num_states_$num_states._train_${act}.log
    done
done
