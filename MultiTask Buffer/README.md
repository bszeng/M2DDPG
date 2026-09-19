# M²DDPG
Pre-collected multi-task training datasets download link: https://pan.baidu.com/s/107xKeWGBTofyOBbnToU22w?pwd=2026


The generated replay buffer files are named as
`replay_buffer_{num_antennas}_{N}_{num_users}_{power_t}_{lr}_{decay}_{num_eps}_{num_time_steps_per_eps}_{buffer_size}.pkl`,
where `N` denotes the number of RIS elements. Four pre-collected buffers with different values of `N` are provided:

| File | N | Size (GB) |
| --- | :---: | :---: |
| `replay_buffer_4_4_4_30_0.001_1e-05_10000_1000_500.pkl` | 4 | 5 |
| `replay_buffer_4_10_4_30_0.001_1e-05_10000_1000_500.pkl` | 10 | 9.2 |
| `replay_buffer_4_20_4_30_0.001_1e-05_10000_1000_500.pkl` | 20 | 16.3 |
| `replay_buffer_4_32_4_30_0.001_1e-05_10000_1000_500.pkl` | 32 | 24.8 |