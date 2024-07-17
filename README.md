# On the Identifiability of Switching Dynamical Systems

[![ArXiv](https://img.shields.io/static/v1?label=arXiv&message=2305.15925&color=B31B1B)](https://arxiv.org/abs/2305.15925) [![Venue:ICML 2024](https://img.shields.io/badge/Venue-ICML_2024-007CFF)](https://icml.cc)  [![Venue:CI4TS 2024](https://img.shields.io/badge/Venue-CI4TS@UAI_24-ebdb34)](https://sites.google.com/view/ci4ts2024/home)

This repository contains source code for the following papers:
- ICML 2024:  [On the identifiability of Switching Dynamical Systems](https://arxiv.org/abs/2305.15925).
- CI4TS Workshop @ UAI 2024:  [Identifying Nonstationary Causal Structures with High-Order Markov Switching Models
](https://arxiv.org/abs/2406.17698).

## Setup

The source code is build on Python 3.8.19 and the models are implemented using Pytorch 2.3. Run `pip install -r requirements.txt` to install the dependencies.


## Testing the code

Below we provide simple indications to test the code.

#### MSM

Generate some synthetic data (defaults to a 2D MSM with 3 states)

``python generate_data_and_train_msm.py --generate_data --no-train --seq_length 100 --num_samples 10000``

Run `python train_msm.py`.

#### SDS

Generate some synthetic data (defaults to a 2D SDS with 2D latent MSM and 3 states)

``python generate_data_and_train_snlds.py --generate_data --no-train --seq_length 200 --num_samples 5000``

Run `python train_snlds.py > log_train_SDS.log`. It is convenient to write the outputs in a separate text file, e.g. `log_train_SDS.log`.

## Results reproduction

To reproduce the results, the files `generate_data_and_train_msm.py` and `generate_data_and_train_snlds.py` can be used.

#### MSM

For example, the following command trains the models to generate the lower left cell of Figure 4b in our paper.

``python generate_data_and_train_msm.py  --seeds 23 24 25 26 27 --dim_obs 3 --num_states 3 --sparsity_prob 0.0 --data_type cosine --device cuda:0 --generate_data --restarts_num 20 --seq_length 200 --num_samples 10000 > log_train_3_states_3_dim.log``

The logs can be found in `log_train_3_states_3_dim.log`. For other settings like polynomials or softplus networks, the ``--data_type`` argument can be set to ``poly`` or ``softplus`` respectively. ``poly`` defaults to cubic polynomials used in the paper, use ``--degreee`` to change the degree type.

#### SDS

The following command trains a Switching Dynamical System on images using different seeds, which can be used to generate the rightmost column of Figure 6 in our paper.

``python generate_data_and_train_snlds.py  --seeds 23 24 25 26 27 --dim_obs 2 --images --dim_latent 2 --num_states 3 --sparsity_prob 0.0 --data_type cosine --device cuda:0 --generate_data --restarts_num 5 --seq_length 200 --num_samples 5000 > log_train_SDS_images.log``

The logs can be found in `log_train_SDS_images.log`, and the models will be saved in ``results/models_sds``. Note the script can take a very long time due to the number of seeds and restarts used. Consider using more restarts for optimal results.

## Misc

For evaluation scripts and trained models, contact the corresponding author.

## Citation

If you use this code, or you find our paper useful for your research, consider citing our paper.
```
@inproceedings{
balsells-rodas2024on,
title={On the Identifiability of Switching Dynamical Systems},
author={Carles Balsells-Rodas and Yixin Wang and Yingzhen Li},
booktitle={Forty-first International Conference on Machine Learning},
year={2024}
}
```
