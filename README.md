<p align="center">
  <h1 align="center">Few-shot Implicit Function Generation via Equivariance</h1>
  <div class="authors">
  <p align="center">
    <strong><a href="https://jeandiable.github.io/">Suizhi Huang<sup>1,2</sup></a></strong>
    &nbsp;&nbsp;
    <strong><a href="https://adamdad.github.io/">Xingyi Yang<sup>2</sup></a></strong>
    &nbsp;&nbsp;
    <strong><a href="https://scholar.google.com/citations?user=GtNuBJcAAAAJ&hl=zh-CN">Hongtao Lu<sup>1</sup></a></strong>
    &nbsp;&nbsp;
    <strong><a href="https://sites.google.com/site/sitexinchaowang/">Xinchao Wang<sup>2</sup></a></strong>
  </p>
</div>

<div class="affiliations">
  <p align="center">
<sup>1</sup>Shanghai Jiao Tong University, <sup>2</sup>National University of Singapore
</p>
  </p>


  <p align="center">
<!--   <a href="https://openaccess.thecvf.com/content/CVPR2024/html/Lu_FedHCA2_Towards_Hetero-Client_Federated_Multi-Task_Learning_CVPR_2024_paper.html"><img alt='cvpr' src="https://img.shields.io/badge/CVPR-2024-blue.svg"></a> -->
  <a href="https://jeandiable.github.io/EquiGen/"><img alt='Website' src="https://img.shields.io/badge/Website-EquiGen-blue"></a>
  <a href="https://arxiv.org/abs/2501.01601"><img alt='arXiv' src="https://img.shields.io/badge/arXiv-2501.01601-b31b1b.svg"></a>
  <!-- add a webpage -->


  </p>
</p>

This is the official implementation of the paper **Few-Shot Implicit Function Generation via Equivariance (CVPR 2025 HIGHLIGHT).**

# ✨ Abstract
Implicit Neural Representations (INRs) have emerged as a powerful framework for representing continuous signals. However, generating diverse INR weights remains challenging due to limited training data. We introduce Few-shot Im- Implicit Function Generation, a new problem setup that aims to generate diverse yet functionally consistent INR weights from only a few examples. This is challenging because even for the same signal, the optimal INRs can vary significantly depending on their initializations. To tackle this, we propose EQUIGEN, a framework that can generate new INRs from limited data. The core idea is that functionally similar networks can be transformed into one another through weight permutations, forming an equivariance group. By projecting these weights into an equivariant latent space, we enable diverse generation within these groups, even with few examples. EQUIGEN implements this through an equivariant encoder trained via contrastive learning and smooth augmentation, an equivariance-guided diffusion process, and controlled perturbations in the equivariant subspace. Experiments on 2D image and 3D shape INR datasets demonstrate that our approach effectively generates diverse INR weights while preserving their functional properties in few-shot scenarios.

# 💡Motivation
![Setting](/docs/assets/setting.gif)

  Illustration of the Few-shot Implicit Function Generation setting with 3D INR data examples. The goal is to generate diverse INR weights from limited target samples. Source samples(top) show previously observed INRs of diverse shape categories. In practice, only limited target samples (bottom left) are available for training. The framework aims to learn a generator that can produce diverse generated samples (right) despite the limited training data. This setting addresses the practical scenario where only a few examples of new shapes are available for training.

# 🔭 Overview
![Overview](/docs/assets/Overview.gif)

The project consists of three main stages:

1. **Pretraining**: Training an equivariant encoder using `pretrain.py`
2. **Training/Fine-tuning**: Diffusion model training guided by the pretrained encoder, implemented in `train.py`. Few-shot fine-tuning is also performed in this stage, controlled by the `is_fewshot` configuration parameter.
3. **Generation**: Generating new implicit functions using `generate.py`

# 🔧 Key Features
- Contrastive learning based equivariant encoder for implicit function weights.
- Smooth augmentation for better feature learning
- Equivariance-guided diffusion with explicit equivariance regularization
- Controlled equivariant subspace pertur- bation for diverse generation
   
# 🏗️ Project Structure

- `pretrain.py` # Equivariant encoder pre-training
- `train.py` # Main training and few-shot fine-tuning, guided by the pretrained encoder
- `generate.py` # Generation of implicit functions
- `models/`
  - `conditional_diffusion.py` # Conditional diffusion model implementation
  - `feature_extractor.py` # equivariant encoder implementation
  - `transformer.py` # denoising diffusion transformer implementation
- `utils/`
  - `info_cse_loss.py` # Loss function for the equivariant encoder
  - `helpers.py` # Helper functions
  - `smooth_augmentation.py` # Smooth augmentation for the weight space
  - `ema.py` # ema model
  - `nn/` #equivariant model implementation
- `data/`
  - `__init__.py`
  - `inr_dataset.py` # Dataset for implicit neural representations
- `configs/`
  - `default.yaml` # Default configuration file
  - `dataset_config.yaml` # Dataset configuration file
- `dataset/`
  - `compute_mnist_statistics.py` # Compute statistics for the MNIST dataset
  - `generate_mnist_data_splits.py` # Generate the dataset splits


# 🚀 Usage

To run the project, use the following command:

0. Install the dependencies:

```shell
pip install -r requirements.txt
```

1. The MNIST-INRs data is available for public, everyone can download it from [here](https://www.dropbox.com/sh/56pakaxe58z29mq/AABtWNkRYroLYe_cE3c90DXVa?dl=0). Please download the data and place it in, e.g. `dataset/mnist-inrs` folder.

2. Create the data split using:

```shell
python generate_data_splits.py --data-path dataset/mnist-inrs
```

This will create a json file `dataset/mnist_splits.json`.

3. Compute the dataset (INRs) statistics using:

```shell
python compute_statistics.py --data-path dataset/mnist_splits.json
```

This will create `dataset/statistics.pth` object.

4. Run the pre-training:

```
python pretrain.py --data_path dataset/mnist_splits.json --n_epochs 500 --batch_size 512 --seed 42 --wandb
```

5. Run the training:

```
accelerate launch train.py
```

Configurations can be changed in `configs/default.yaml` and together with the dataset configuration in `configs/dataset_config.yaml`.

6. Run the generation:

```
accelerate launch generate.py --test_name test_name
```

## Evaluation

To evaluate the model, use the following command for the LPIPS score:

```
python evaluate_lpips.py --folder1 path/to/folder/with/images
```

For the FID score, please follow the official [pytorch-fid](https://github.com/mseitzer/pytorch-fid) implementation. 


# 🗓 Plan
- [x] MNIST-INR Support
- [ ] CIFAR-10-INR Support
- [ ] ShapeNet-INR Support
- [ ] ...

# Acknowledgement
This project is build upon several previous works, including [Neumeta](https://github.com/Adamdad/neumeta), [DWSNet](https://github.com/AvivNavon/DWSNets), [HyperDiffusion](https://github.com/Rgtemze/HyperDiffusion), and several commonly used repos including [Diffusers](https://github.com/huggingface/diffusers), [Accelerator](https://github.com/huggingface/accelerate) and so on. Show our deepest respect for them.

# 📖 Citation
```
@article{huang2025few,
  title={Few-shot Implicit Function Generation via Equivariance},
  author={Huang, Suizhi and Yang, Xingyi and Lu, Hongtao and Wang, Xinchao},
  journal={arXiv preprint arXiv:2501.01601},
  year={2025}
}
```
