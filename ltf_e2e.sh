#!/usr/bin/zsh


NAVSIM_PATH=/nas/users/hyzhou/PAMI2024/release/navsim
cd ${NAVSIM_PATH}
echo ${PWD}
CUDA_VISIBLE_DEVICES=${1} pixi run python ltf_e2e.py output=$2
cd -
