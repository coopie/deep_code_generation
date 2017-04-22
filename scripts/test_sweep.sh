#!/bin/sh

COMMAND='python experiments/Recurrent_VAE_baseline/experiment_128k.py'

$COMMAND single_layer_gru_8
./scripts/gdrive upload experiments/Recurrent_VAE_baseline/single_layer_gru_8

echo "First experiment finished" | mail -s 'Masters Project Bot' sam.j.coope@gmail.com
