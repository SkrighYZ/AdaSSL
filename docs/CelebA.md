
# Fine-grained attribute classification on CelebA

## Data

Please set `$DATA_DIR` to the parent directory of the datasets. 

First, unzip the files we provide to `$DATA_DIR`. It should create a folder `$DATA_DIR/celeba` with several files. Inside, we have the annotation files `{train,val,test}.txt`. Train and val/test splits have disjoint celebrity identities (20% of randomly-selected identities go to val/test). Val and test share the same identities and the data examples are selected randomly. Instead of searching for natural image pairs on the fly, we generate all pairs of images for each identity in the training set and store them in `pairs_train.npy`. 

```bash
unzip "${ROOT_DIR}/AdaSSL/metadata/celeba.zip" -d "${DATA_DIR}/"
```

Then, download the cropped and aligned version of [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) from [this link](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ) and extract its content into `$DATA_DIR/celeba/img_align_celeba/`.

You should get a directory with structure like the following.

```bash
$DATA_DIR/
├── ...
├── celeba/
│   ├── img_align_celeba/
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── ...
│   ├── attributes_test.txt
│   ├── attributes_train.txt
│   ├── attributes_val.txt
│   └── pairs_train.npy
└── ...
```

## Training

Please set `$save_dir` to the directory where you'd like to store the checkpoints. You need to use a different `$save_dir` for each run. Note that we do not log to this directory, as logging is done through *wandb*.

We use SimCLR or BYOL as our SSL framework controlled by `--ssl_objective={simclr|byol}`. Here are example commands to run experiments with SimCLR on natural pairs with strong augmentations. To use weak augmentations, set `--transform_type=weak`.
For experiments using standard pairs, add the `--use_standard_pairing` flag. 

Note that you might need to change the hyperparameters for different settings; we provide them in Sec. E.4 of the paper.


### Vanilla

```bash
python "${ROOT_DIR}/AdaSSL/train.py" \
	--dataset celeba \
	--input_size 64 \
	--ssl_objective simclr \
	--model vanilla \
	--projector 512-1024-128 \
	--transform_type strong \
	--data_dir "${DATA_DIR}/celeba" \
	--save_dir "$save_dir" \
	--train_steps 80000 \
	--log_steps 100 \
	--save_steps 20000 \
	--eval_steps 5000 \
	--batch_size 512 \
	--eval_batch_size 512 \
	--learning_rate 2e-4 \
	--weight_decay 1e-4 \
	--tau 0.1 \
	--learnable_lambda \
	--num_workers 12 \
	--resume \
	--amp  
```

### AdaSSL-V

```bash
python "${ROOT_DIR}/AdaSSL/train.py" \
	--dataset celeba \
	--input_size 64 \
	--ssl_objective simclr \
	--model adassl-v \
	--additional_view \
	--projector 512-1024-128 \
	--latent_predictor 128-1024-20 \
	--editor 148-512-128 \
	--reg_beta 0.1 \
	--reg_beta_warmup_steps 10000 \
	--transform_type strong \
	--data_dir "${DATA_DIR}/celeba" \
	--save_dir "$save_dir" \
	--train_steps 80000 \
	--log_steps 100 \
	--save_steps 20000 \
	--eval_steps 5000 \
	--batch_size 512 \
	--eval_batch_size 512 \
	--learning_rate 2e-4 \
	--weight_decay 1e-4 \
	--tau 0.1 \
	--learnable_lambda \
	--num_workers 12 \
	--resume \
	--amp  
```

### AdaSSL-S

```bash
python "${ROOT_DIR}/AdaSSL/train.py" \
	--dataset celeba \
	--input_size 64 \
	--ssl_objective simclr \
	--model adassl-s \
	--additional_view \
	--projector 512-1024-128 \
	--latent_predictor 128-1024-20 \
	--reg_beta 0.5 \
	--reg_beta_warmup_steps 0 \
	--transform_type strong \
	--data_dir "${DATA_DIR}/celeba" \
	--save_dir "$save_dir" \
	--train_steps 80000 \
	--log_steps 100 \
	--save_steps 20000 \
	--eval_steps 5000 \
	--batch_size 512 \
	--eval_batch_size 512 \
	--learning_rate 2e-4 \
	--weight_decay 1e-4 \
	--tau 0.1 \
	--learnable_lambda \
	--num_workers 12 \
	--resume \
	--amp  
```


## Evaluation

After training, run the following script to train offline linear probes on the train set and evaluate them on the test set. The code needs the encoder and projector of the final checkpoint (`resnet.pt` and `projector.pt`).

```bash
python "${ROOT_DIR}/AdaSSL/linear_eval.py" \
    --data_dir "${DATA_DIR}/celeba" \
    --save_dir "$save_dir" \
    --train_steps 50000 \
	--eval_steps 2000 \
	--log_steps 100 \
	--batch_size 512 \
	--eval_batch_size 128 \
	--learning_rate 2e-4 \
	--weight_decay 1e-4 \
	--num_workers 4
```
