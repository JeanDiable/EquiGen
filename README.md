<p align="center">
  <h1 align="center">Few-shot Implicit Function Generation via Equivariance</h1>
  <p align="center">
    <strong><a href="https://jeandiable.github.io/">Suizhi Huang</a></strong>
    &nbsp;&nbsp;
    <strong><a href="https://adamdad.github.io/">Xingyi Yang</a></strong>
    &nbsp;&nbsp;
    <strong><a href="https://scholar.google.com/citations?user=GtNuBJcAAAAJ&hl=zh-CN">Hongtao Lu</a></strong>
    &nbsp;&nbsp;
    <strong><a href="https://sites.google.com/site/sitexinchaowang/">Xinchao Wang</a></strong>
  </p>

  <p align="center">
<!--   <a href="https://openaccess.thecvf.com/content/CVPR2024/html/Lu_FedHCA2_Towards_Hetero-Client_Federated_Multi-Task_Learning_CVPR_2024_paper.html"><img alt='cvpr' src="https://img.shields.io/badge/CVPR-2024-blue.svg"></a> -->
  <a href="https://arxiv.org/abs/2501.01601"><img alt='arXiv' src="https://img.shields.io/badge/arXiv-2311.13250-b31b1b.svg"></a>
  </p>
</p>

![Overview](assets/overview.png)

## ✨ Abstract
Implicit Neural Representations (INRs) have emerged as a powerful framework for representing continuous signals. However, generating diverse INR weights remains challenging due to limited training data. We introduce Few-shot Im- Implicit Function Generation, a new problem setup that aims to generate diverse yet functionally consistent INR weights from only a few examples. This is challenging because even for the same signal, the optimal INRs can vary significantly depending on their initializations. To tackle this, we propose EQUIGEN, a framework that can generate new INRs from limited data. The core idea is that functionally similar networks can be transformed into one another through weight permutations, forming an equivariance group. By projecting these weights into an equivariant latent space, we enable diverse generation within these groups, even with few examples. EQUIGEN implements this through an equivariant encoder trained via contrastive learning and smooth augmentation, an equivariance-guided diffusion process, and controlled perturbations in the equivariant subspace. Experiments on 2D image and 3D shape INR datasets demonstrate that our approach effectively generates diverse INR weights while preserving their functional properties in few-shot scenarios.

## Code

The code will be made public in the near future.


## 📖 Citation
```
@article{huang2025few,
  title={Few-shot Implicit Function Generation via Equivariance},
  author={Huang, Suizhi and Yang, Xingyi and Lu, Hongtao and Wang, Xinchao},
  journal={arXiv preprint arXiv:2501.01601},
  year={2025}
}
```
