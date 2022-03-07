#!/bin/bash

detection='negative'
num_set='A'
version='version2'

. utils/parse_options.sh

model="Transformer"
config_path=conf/${model}/${detection}/
config_name="${model}_ver2.yaml"

train_manifest="manifest/${detection}/${num_set}/train_sp.json"
val_manifest="manifest/${detection}/${num_set}/dev.json"
test_manifest="manifest/${detection}/${num_set}/test.json"
exp_dir=logs/${model}/${detection}

rm -r $exp_dir/$num_set/$version/*

python3 ./train.py --config-path=$config_path --config-name=$config_name \
    trainer.max_epochs=200  \
    model.train_ds.batch_size=32 model.validation_ds.batch_size=32 \
    model.train_ds.manifest_filepath=${train_manifest} \
    model.validation_ds.manifest_filepath=${val_manifest} \
    model.test_ds.manifest_filepath=${test_manifest} \
    trainer.gpus=2 \
    exp_manager.name=${num_set} +exp_manager.use_datetime_version=False \
    +exp_manager.version=${version} \
    exp_manager.exp_dir=${exp_dir}
