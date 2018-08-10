#!/bin/bash

dropbox=../../dropbox

grammar_file=$dropbox/context_free_grammars/mol_zinc.grammar
sk=0
#data_dump=$dropbox/data/zinc/250k_rndm_zinc_drugs_clean-${sk}.h5
data_dump=$dropbox/data/mattrack_avail/mattrack_mostly_good_subset_1260K-0.h5
prop_file=$dropbox/data/mattrack_avail/mattrack_mostly_good_subset_1260K-0_xlogp.csv
#data_dump=$dropbox/data/mattrack_avail/10_glp1_plus_mattrack-${sk}.h5
#data_dump=$dropbox/data/mattrack_avail/05_glp1_plus_mattrack-${sk}.h5
#data_dump=$dropbox/data/glp1/glp1_smiles-0.h5
#data_dump=$dropbox/data/zinc_01/250k_rndm_zinc_drugs_clean-${sk}.h5

bsize=300
enc=cnn
ae_type=vae
loss_type=vanilla
rnn_type=gru
kl_coeff=1
lr=0.0001
num_epochs=100
eps_std=0.01
prob_fix=0
save_dir=/localscratch/sdvae_results/graph_generation/mattrack_prop_01_vanilla-sk-${sk}-mol_${ae_type}/enc-${enc}-loss-${loss_type}-eps-${eps_std}-rnn-${rnn_type}-kl-${kl_coeff}

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

export CUDA_VISIBLE_DEVICES=1
python train_zinc_prop.py \
    -grammar_file $grammar_file \
    -data_dump $data_dump \
    -prop_file $prop_file \
    -old $old \
    -batch_size $bsize \
    -encoder_type $enc \
    -save_dir $save_dir \
    -ae_type $ae_type \
    -learning_rate $lr \
    -rnn_type $rnn_type \
    -num_epochs $num_epochs \
    -eps_std $eps_std \
    -loss_type $loss_type \
    -kl_coeff $kl_coeff \
    -prob_fix $prob_fix \
    -mode gpu
