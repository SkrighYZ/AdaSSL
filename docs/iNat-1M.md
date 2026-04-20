
# Fine-grained species classification on iNat-1M


## Data

Please set `$DATA_DIR` to the parent directory of the datasets. 

First, unzip the files we provide to `$DATA_DIR`. It should create a folder `$DATA_DIR/inat` with several files. The category and super-category labels are stored in `{train,val,test}.txt`. We also provide some mappings that come in handy when we create image pairs. `cat_to_indices.pkl` maps each class to the indices of training images in that class. `super_to_indices.pkl` maps each superclass to the indices of training images in that superclass.

```bash
unzip "${ROOT_DIR}/AdaSSL/metadata/inat.zip" -d "${DATA_DIR}/"
```

Then, download the official iNat21 training set [here](https://github.com/visipedia/inat_comp/tree/master/2021) (224GB). We only need `train.tar.gz`. iNat-1M is a class-balanced subset of the iNat21 training set, with 5000 species, each has 200 training images, 50 validation images, and 50 test images. 

```bash
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train.tar.gz -P "${DATA_DIR}/inat"
tar -xf "${DATA_DIR}/inat/train.tar.gz" -C "${DATA_DIR}/inat"
```

This should generate a directory with structure like the following.

```bash
$DATA_DIR/
├── ...
├── inat/
│   ├── train.tar.gz
│   ├── train/
│   │   ├── 00000_Animalia_Annelida_Clitellata_Haplotaxida_Lumbricidae_Lumbricus_terrestris/
│   │   │   ├── 5d4c63dc-da66-4193-aa22-e97e5191ef25.jpg
│   │   │   └── ...
│   │   ├── 00005_Animalia_Arthropoda_Arachnida_Araneae_Antrodiaetidae_Atypoides_riversi/
│   │   │   ├── 836c0cc8-f737-43e1-86d2-c1551d660e9b.jpg
│   │   │   └── ...
│   │   └── ...
│   ├── test.txt
│   ├── train.txt
│   ├── val.txt
│   ├── cat_to_indices.pkl 
│   ├── super_to_indices.pkl 
│   └── label_types_n{0.0,0.25,0.5,0.75,1.0}.txt
└── ...
```


## Training

Please set `$save_dir` to the directory where you'd like to store the checkpoints. You need to use a different `$save_dir` for each run. Note that we do not log to this directory, as logging is done through *wandb*.

In a perfect supervised learning scenario, we would pair up images from the same species all the time so that the model clusters their embeddings together. Here, we randomly corrupt a portion of these pairings, such that they are grouped by coarse, superclass labels instead.

First, we need to select a noise ratio between 0 to 1, which indicates how many of the data pairs seen by the model are corrupted. When this ratio is 0, the model is trained only on same-species image pairs, effectively turning the problem into supervised learning. Then, we generate a file that randomly assigns whether an image in the training set is using fine- or coarse-grained pairing based on this noise ratio.

We provide these files we use for our experiments in `${ROOT_DIR}/AdaSSL/metadata/inat.zip`. You can also easily generate a new one like the following.

```bash
noise_ratio="0.75"
awk -v n=1000000 -v p="$noise_ratio" 'BEGIN {srand(); for(i=1; i<=n; i++) print (rand() < p ? 1 : 0)}' > "${DATA_DIR}/inat/label_types_n${noise_ratio}.txt"
```

We compare the vanilla SimCLR and AdaSSL-V in this experiment under different `noise_ratio`. We should see larger improvements from AdaSSL-V under higher noise. Here are the example scripts.


### Vanilla

```bash
python "${ROOT_DIR}/AdaSSL/train.py" \
        --dataset inat \
        --input_size 224 \
        --ssl_objective simclr \
        --backbone resnet50 \
        --tau 0.1 \
        --learnable_lambda \
        --noise_ratio "$noise_ratio" \
        --model vanilla \
        --projector 2048-1024-128 \
        --transform_type strong \
        --data_dir "${DATA_DIR}/inat" \
        --save_dir "$save_dir" \
        --train_steps 200000 \
        --log_steps 100 \
        --save_steps 20000 \
        --eval_steps 20000 \
        --batch_size 256 \
        --eval_batch_size 128 \
        --learning_rate 2e-4 \
        --weight_decay 1e-4 \
        --num_workers 10 \
        --resume \
        --amp
```


### AdaSSL-V

```bash
python "${ROOT_DIR}/AdaSSL/train.py" \
        --dataset inat \
        --input_size 224 \
        --ssl_objective simclr \
        --backbone resnet50 \
        --tau 0.1 \
        --learnable_lambda \
        --noise_ratio "$noise_ratio" \
        --model adassl-v \
        --additional_view \
        --projector 2048-1024-128 \
        --latent_predictor 128-1024-8 \
        --editor 136-1024-128 \
        --reg_beta 0.8 \
        --reg_beta_warmup_steps 10000 \
        --transform_type strong \
        --data_dir "${DATA_DIR}/inat" \
        --save_dir "$save_dir" \
        --train_steps 200000 \
        --log_steps 100 \
        --save_steps 20000 \
        --eval_steps 20000 \
        --batch_size 256 \
        --eval_batch_size 128 \
        --learning_rate 2e-4 \
        --weight_decay 1e-4 \
        --num_workers 10 \
        --resume \
        --amp
```


## Evaluation

We log the online linear probes' performance to *wandb*.