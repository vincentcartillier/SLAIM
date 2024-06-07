# SLAIM

Official code release for the paper:

> **[SLAIM: Robust Dense Neural SLAM for Online Tracking and Mapping]** <br />
> *[Vincent Cartillier](https://vincentcartillier.github.io/), Grant Schindler, Irfan Essa* <br />
> Neural Rendering Intelligence workshop CVPR 2024 <br />


[[Project page](https://vincentcartillier.github.io/slaim.html)], [[arXiv](https://arxiv.org/abs/2404.11419)]

![High level overview of SLAIM capabilities](assets/slaim.png)


```bibtex
@inproceedings{cartillier2024slaim_cvpr_nri_24,
  title={SLAIM: Robust Dense Neural SLAM for Online Tracking and Mapping},
  author={Cartillier, Vincent and Schindler, Grant and Essa, Irfan},
  booktitle = {CVPR(NRI)},
  year={2024}
}

@article{cartillier2024slaim,
  title={SLAIM: Robust Dense Neural SLAM for Online Tracking and Mapping},
  author={Cartillier, Vincent and Schindler, Grant and Essa, Irfan},
  journal={arXiv preprint arXiv:2404.11419},
  year={2024}
}
```

## Setup
1. Clone SLAIM
```bash
git clone --recursive https://github.com/vincentcartillier/SLAIM
cd SLAIM
# git submodule update --init --recursive
```

2. Create a conda environment,
```bash
conda create -n slaim python=3.11 cmake=3.14.0
conda activate slaim
pip install -r requirements.txt
```

3. Install Instant-NGP.
```bash
cd dependencies/Instant-NGP
```

## Demo



## Data



## Workflow


