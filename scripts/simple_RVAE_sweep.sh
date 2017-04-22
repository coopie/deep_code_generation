#!/bin/sh

COMMAND='python experiments/Recurrent_VAE_baseline/experiment_128k.py'

$COMMAND single_layer_gru_8
./scripts/gdrive upload experiments/Recurrent_VAE_baseline/single_layer_gru_8

echo "First experiment finished" | mail -s 'Masters Project Bot' sam.j.coope@gmail.com
$COMMAND single_layer_gru_16
./scripts/gdrive upload experiments/Recurrent_VAE_baseline/single_layer_gru_16
$COMMAND single_layer_gru_32
./scripts/gdrive upload experiments/Recurrent_VAE_baseline/single_layer_gru_32
$COMMAND single_layer_gru_64
./scripts/gdrive upload experiments/Recurrent_VAE_baseline/single_layer_gru_64
$COMMAND single_layer_gru_128
./scripts/gdrive upload experiments/Recurrent_VAE_baseline/single_layer_gru_128
$COMMAND single_layer_gru_256
./scripts/gdrive upload experiments/Recurrent_VAE_baseline/single_layer_gru_256

echo "Experiments finished." | mail -s 'Masters Project Bot' sam.j.coope@gmail.com
