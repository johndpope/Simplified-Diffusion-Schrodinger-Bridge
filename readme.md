# Simplified Diffusion Schrödinger Bridge

This is the official implementation of the paper [Simplified Diffusion Schrödinger Bridge](https://arxiv.org/abs/2403.14623).

**Abstract**
> This paper introduces a novel theoretical simplification of the Diffusion Schrödinger Bridge (DSB) that facilitates its unification with Score-based Generative Models (SGMs), addressing the limitations of DSB in complex data generation and enabling faster convergence and enhanced performance. By employing SGMs as an initial solution for DSB, our approach capitalizes on the strengths of both frameworks, ensuring a more efficient training process and improving the performance of SGM. We also propose a reparameterization technique that, despite theoretical approximations, practically improves the network's fitting capabilities. Our extensive experimental evaluations confirm the effectiveness of the simplified DSB, demonstrating its significant improvements. We believe the contributions of this work pave the way for advanced generative modeling.

## Installation

1. Clone the repo
   
   ```bash
   git clone https://github.com/tzco/Simplified-Diffusion-Schrodinger-Bridge.git
   cd Simplified-Diffusion-Schrodinger-Bridge
   ```

2. Setup conda environment
   
   ```bash
   conda create -n sdsb python=3.10 -y
   conda activate sdsb

   # install torch first, here is an example for cuda 11.8
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

   # install required packages
   pip install -r requirements.txt
   ```

3. Prepare dataset

   Download the [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [AFHQ](https://github.com/clovaai/stargan-v2) datasets into the folder `dataset`.

4. Download checkpoints

   We provide pretrained checkpoints [AFHQ256](https://github.com/TiankaiHang/storage-2023/releases/download/sdsb/afhq256.pth), [AFHQ512](https://github.com/TiankaiHang/storage-2023/releases/download/sdsb/afhq512.pth), and [CelebA](https://github.com/TiankaiHang/storage-2023/releases/download/sdsb/celeba.pth) for inference.

   We also provide Flow Matching models [AFHQ256_pretrain](https://github.com/TiankaiHang/storage-2023/releases/download/sdsb/afhq256_pretrain.pth) and [AFHQ512_pretrain](https://github.com/TiankaiHang/storage-2023/releases/download/sdsb/afhq512_pretrain.pth) for initialization.

   Download them into the folder `ckpt`, or you can also download with [`bash script/download_checkpoint.sh`](./script/download_checkpoint.sh).

## Inference

Here we provide some example scripts for sampling from pre-trained models.

**AFHQ 512**

```bash
python inference.py --network adm --prior afhq-dog-512 \
   --dataset afhq-cat-512 --simplify --reparam term \
   --gamma_type linear_1e-3_1e-2 --exp_name trdsb-afhq512 \
   --ckpt ./ckpt/afhq512.pth --num_sample 128 \
   --batch_size 16
```

`--prior` sets the prior distribution ($p_{\text{prior}}$); `--dataset` is the data distribution ($p_{\text{data}}$); `--simplify` is a flag to use *Simplified DSB*; `--reparam` chooses the way for reparameterization, `term`
 means Terminal Reparameterization, `flow` means Flow Reparameterization, default is `None`; `--gamma_type` controls the way to add noise to construct $p_{\text{ref}}$; `--ckpt` points to the path of pre-trained model.

Or you could run `python inference.py -h` to see the full argument list.

**AFHQ 256**

```bash
python inference.py --network adm --prior afhq-dog-256 \
   --dataset afhq-cat-256 --simplify --reparam term \
   --gamma_type linear_1e-3_1e-2 --exp_name trdsb-afhq256 \
   --ckpt ./ckpt/afhq256.pth
```

**CelebA**

```bash
python inference.py --network uvit-b --prior pixel-standard \
   --dataset celeba-64 --simplify --reparam term \
   --gamma_type linear_1e-5_1e-4 --exp_name trdsb-celeba \
   --ckpt ./ckpt/celeba.pth
```

## Training

**2D experiments**

```bash
# Original DSB
torchrun --standalone train.py \
   --exp2d \                              # to train on 2D datasets
   --method dsb \                         # use DSB method
   --noiser dsb \                         # use DSB p_ref
   --prior dsb-pinwheel \                 # prior distribution (p_prior)
   --dataset checkerboard:8 \             # data distribution (p_data)
   --training_timesteps 16 \              # timesteps for training
   --inference_timesteps 16 \             # timesteps for inference
   --gamma_type linear_1e-4_1e-3 \        # gamma schedule, control the way to add noise
   --repeat_per_epoch 8 \                 # multiplier of iterations per epoch
   --epochs 41 \                          # total training epochs
   --exp_name dsb-pinwheel-checkerboard   # name of experiment

# Simplified DSB
torchrun --standalone train.py --exp2d --method dsb --prior dsb-pinwheel --dataset checkerboard:8 --training_timesteps 16 --inference_timesteps 16 --gamma_type linear_1e-4_1e-3 --repeat_per_epoch 8 --epochs 41 --exp_name sdsb-pinwheel-checkerboard \
   --noiser flow \                        # use Flow Matching p_ref
   --simplify                             # use Simplified DSB

# Terminal Reparameterized DSB
torchrun --standalone train.py --exp2d --method dsb --prior dsb-pinwheel --dataset checkerboard:8 --training_timesteps 16 --inference_timesteps 16 --gamma_type linear_1e-4_1e-3 --repeat_per_epoch 8 --epochs 41 --exp_name trdsb-pinwheel-checkerboard \
   --noiser flow \                        # use Flow Matching p_ref
   --simplify \                           # use Simplified DSB
   --reparam term                         # use Terminal Reparameterization

# Flow Reparameterized DSB
torchrun --standalone train.py --exp2d --method dsb --prior dsb-pinwheel --dataset checkerboard:8 --training_timesteps 16 --inference_timesteps 16 --gamma_type linear_1e-4_1e-3 --repeat_per_epoch 8 --epochs 41 --exp_name frdsb-pinwheel-checkerboard \
   --noiser flow \                        # use Flow Matching p_ref
   --simplify \                           # use Simplified DSB
   --reparam flow                         # use Flow Reparameterization
```

**AFHQ 256**

To train from scratch, run

```bash
torchrun --standalone --nproc_per_node=8 train.py --method dsb --noiser flow --network adm --batch_size 192 --prior afhq-dog-256 --dataset afhq-cat-256 --val_prior afhq-dog-256 --val_data afhq-cat-256 --lr 1e-5 --repeat_per_epoch 256 --use_amp --training_timesteps 100 --inference_timesteps 100 --simplify --reparam term --gamma_type linear_1e-3_1e-2 --exp_name trdsb-afhq256
```

To train with Flow Matching models as initialization, run

```bash
torchrun --standalone --nproc_per_node=8 train.py --method dsb --noiser flow --network adm --batch_size 192 --prior afhq-dog-256 --dataset afhq-cat-256 --val_prior afhq-dog-256 --val_data afhq-cat-256 --lr 1e-5 --repeat_per_epoch 256 --use_amp --training_timesteps 100 --inference_timesteps 100 --simplify --reparam term --gamma_type linear_1e-3_1e-2 --exp_name trdsb-afhq256 \
   --ckpt ./ckpt/afhq256_pretrain.pth --skip_epochs 1
```

**AFHQ 512**

To train from scratch, run

```bash
torchrun --standalone --nproc_per_node=8 train.py --method dsb --noiser flow --network adm --batch_size 192 --prior afhq-dog-512 --dataset afhq-cat-512 --val_prior afhq-dog-512 --val_data afhq-cat-512 --lr 1e-5 --repeat_per_epoch 256 --use_amp --training_timesteps 100 --inference_timesteps 100 --simplify --reparam term --gamma_type linear_1e-3_1e-2 --exp_name trdsb-afhq512
```

To train with Flow Matching models as initialization, run

```bash
torchrun --standalone --nproc_per_node=8 train.py --method dsb --noiser flow --network adm --batch_size 192 --prior afhq-dog-512 --dataset afhq-cat-512 --val_prior afhq-dog-512 --val_data afhq-cat-512 --lr 1e-5 --repeat_per_epoch 256 --use_amp --training_timesteps 100 --inference_timesteps 100 --simplify --reparam term --gamma_type linear_1e-3_1e-2 --exp_name trdsb-afhq512 \
   --ckpt ./ckpt/afhq512_pretrain.pth --skip_epochs 1
```

If you want to accelerate the training with multiple nodes, e.g., 4, you could replace `torchrun --standalone --nproc_per_node=8` with `torchrun --nnodes=4 --nproc_per_node=8 --max_restarts=3 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT --node_rank=$NODE_RANK`, where `MASTER_ADDR`, `MASTER_PORT`, and `NODE_RANK` are distributed training related environment variables.

## Citation
If you find our work useful for your research, please consider citing our paper.
```
@misc{tang2024simplified,
    title={Simplified Diffusion Schrödinger Bridge},
    author={Zhicong Tang and Tiankai Hang and Shuyang Gu and Dong Chen and Baining Guo},
    year={2024},
    eprint={2403.14623},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
