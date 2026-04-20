
# Identifying data-generating factors on 3DIdent 

## Data

Please set `$DATA_DIR` to the parent directory of the datasets. Create the 3DIdent folder `$DATA_DIR/3dident`.

Download 3DIdent from [the official dataset page](https://zenodo.org/records/4502485). 

```bash
wget https://zenodo.org/records/4502485/files/3dident_train.tar?download=1 -P "${DATA_DIR}/3dident"
tar -xf "${DATA_DIR}/3dident/3dident_train.tar" -C "$DATA_DIR"
wget https://zenodo.org/records/4502485/files/3dident_test.tar?download=1 -P "${DATA_DIR}/3dident"
tar -xf "${DATA_DIR}/3dident/3dident_test.tar" -C "$DATA_DIR"
```

This should generate a directory with structure like the following.

```bash
$DATA_DIR/
├── ...
├── 3dident/
│   ├── test/
│   │   ├── images/
│   │   │   ├── 00000.png
│   │   │   ├── 00001.png
│   │   │   └── ...
│   │   ├── latents.npy
│   │   └── raw_latents.npy
│   └── train/
│       ├── images/
│       │   ├── 00000.png
│       │   ├── 00001.png
│       │   └── ...
│       ├── latents.npy
│       └── raw_latents.npy
└── ...
```

## Training

Please set `$save_dir` to the directory where you'd like to store the checkpoints. You need to use a different `$save_dir` for each run. Note that we do not log to this directory, as logging is done through *wandb*.

Our experiments in the paper using natural pairs have the following configurations. We use SimCLR as our SSL framework, including its InfoNCE objective and projector architecture.


### Vanilla

```bash
python "${ROOT_DIR}/AdaSSL/train.py" \
	--dataset 3dident \
	--input_size 128 \
	--ssl_objective simclr \
	--model vanilla \
	--projector 512-128-16 \
	--transform_type weak \
	--data_dir "${DATA_DIR}/3dident" \
	--save_dir "$save_dir" \
	--train_steps 150000 \
	--log_steps 100 \
	--save_steps 50000 \
	--eval_steps 5000 \
	--batch_size 256 \
	--eval_batch_size 256 \
	--learning_rate 1e-4 \
	--weight_decay 1e-5 \
	--tau 0.05 \
	--num_workers 8 \
	--resume \
	--amp  
```

### AdaSSL-V

```bash
python "${ROOT_DIR}/AdaSSL/train.py" \
	--dataset 3dident \
	--input_size 128 \
	--ssl_objective simclr \
	--model adassl-v \
	--projector 512-128-16 \
	--latent_predictor 16-128-128-16 \
	--editor 32-16 \	# For additive editor, just put a number here like "--editor 16". The additive editor achieves a much higher DCI-D score.
	--reg_beta 0.5 \
	--reg_beta_warmup_steps 10000 \
	--transform_type weak \
	--data_dir "${DATA_DIR}/3dident" \
	--save_dir "$save_dir" \
	--train_steps 150000 \
	--log_steps 100 \
	--save_steps 50000 \
	--eval_steps 5000 \
	--batch_size 256 \
	--eval_batch_size 256 \
	--learning_rate 1e-4 \
	--weight_decay 1e-5 \
	--tau 0.05 \
	--num_workers 8 \
	--resume \
	--amp  
```

### AdaSSL-S

```bash
python "${ROOT_DIR}/AdaSSL/train.py" \
	--dataset 3dident \
	--input_size 128 \
	--ssl_objective simclr \
	--model adassl-s \
	--projector 512-128-16 \
	--latent_predictor 16-128-128-16 \
	--reg_beta 0.5 \
	--reg_beta_warmup_steps 0 \
	--transform_type weak \
	--data_dir "${DATA_DIR}/3dident" \
	--save_dir "$save_dir" \
	--train_steps 150000 \
	--log_steps 100 \
	--save_steps 50000 \
	--eval_steps 5000 \
	--batch_size 256 \
	--eval_batch_size 256 \
	--learning_rate 1e-4 \
	--weight_decay 1e-5 \
	--tau 0.05 \
	--num_workers 8 \
	--resume \
	--amp  
```


## Evaluation

After training, run the following script to obtain the *DCI disentanglement* and *linear regression* $R^2$ scores on the test set on top of the frozen embeddings of the final checkpoint (it needs `resnet.pt` and `projector.pt`).


```bash
python "${ROOT_DIR}/AdaSSL/linear_eval.py" \
    --data_dir "${DATA_DIR}/3dident" \
    --save_dir "$save_dir" \
	--batch_size 256 \
	--eval_batch_size 256 \
	--num_workers 4 
```

