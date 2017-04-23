#!/bin/sh

COMMAND='python experiments/Recurrent_VAE_baseline/experiment_128k.py'

EXPERIMENT=single_layer_gru_8
$COMMAND $EXPERIMENT
./scripts/gdrive upload experiments/Recurrent_VAE_baseline/$EXPERIMENT
echo "experiment ${EXPERIMENT} finished" | mail -s 'Masters Project Bot' sam.j.coope@gmail.com

EXPERIMENT=single_layer_gru_16
$COMMAND $EXPERIMENT
./scripts/gdrive upload experiments/Recurrent_VAE_baseline/$EXPERIMENT
echo "experiment ${EXPERIMENT} finished" | mail -s 'Masters Project Bot' sam.j.coope@gmail.com

EXPERIMENT=single_layer_gru_32
$COMMAND $EXPERIMENT
./scripts/gdrive upload experiments/Recurrent_VAE_baseline/$EXPERIMENT
echo "experiment ${EXPERIMENT} finished" | mail -s 'Masters Project Bot' sam.j.coope@gmail.com


EXPERIMENT=single_layer_gru_64
$COMMAND $EXPERIMENT
./scripts/gdrive upload experiments/Recurrent_VAE_baseline/$EXPERIMENT
echo "experiment ${EXPERIMENT} finished" | mail -s 'Masters Project Bot' sam.j.coope@gmail.com

EXPERIMENT=single_layer_gru_128
$COMMAND $EXPERIMENT
./scripts/gdrive upload experiments/Recurrent_VAE_baseline/$EXPERIMENT
echo "experiment ${EXPERIMENT} finished" | mail -s 'Masters Project Bot' sam.j.coope@gmail.com

EXPERIMENT=single_layer_gru_256
$COMMAND $EXPERIMENT
./scripts/gdrive upload experiments/Recurrent_VAE_baseline/$EXPERIMENT
echo "experiment ${EXPERIMENT} finished" | mail -s 'Masters Project Bot' sam.j.coope@gmail.com
