<p align="center">
  <img src="logo.png" alt="Logo of LmPT" width="500">
</p>

**MorphoSkel3D** is based on morphology to facilitate an efficient skeletonization of shapes.

This repository includes the methods as introduced in **MorphoSkel3D: Morphological Skeletonization of 3D Point Clouds for Informed Sampling in Object Classification and Retrieval** &rarr; [ [arXiv](https://arxiv.org/abs/2501.12974) ].

## Skeletonization
The pre-processed skeletons are available for download via [MorphoSkel3D](https://drive.google.com/drive/folders/1n7AdNwgjQS8XOvmVVuXuVT2pQybz3RXr?usp=sharing).  
Place the *shapenet* and *modelnet* content under the `data` directory.

The **ShapeNet** skeletons can be generated from scratch for a specific category:
```bash
python evaluation/shapenet_time.py --category Airplane --tag Demo
```

The **ModelNet** skeletons can be generated from scratch for a specific split:
```bash
python method/modelnet_MS3D.py --split train --tag Demo
```

## Quick  Start
To clone this repository with its submodule of [Skeleton-Aware-Sampling](https://github.com/Pierreoo/Skeleton-Aware-Sampling/tree/61c8968e5915d7b7d1afd982c48054d265fcbf96):
```
git clone --recursive https://github.com/Pierreoo/MorphoSkel3D.git
```

### Pre-trained models
The pre-trained models for the four sampling rations are available via [MorphoSkel3D](https://drive.google.com/drive/folders/1n7AdNwgjQS8XOvmVVuXuVT2pQybz3RXr?usp=sharing).  
Place the *log* content under the `Skeleton-Aware-Sampling/sampling/log` directory.


### Skeleton Extraction
The reconstruction error from skeletal spheres:
```bash
python evaluation/shapenet_recon.py --category Airplane --evaluation recon --distance chamfer
```

### Object Classification
The classification accuracy under a sampling ratio:
```bash
cd Skeleton-Aware-Sampling
python classifi.py --test --ratio 64
```

### Point Cloud Retrieval
The retrieval precision under a sampling ratio:
```bash
cd Skeleton-Aware-Sampling
python retrieval.py --ratio 64
```

## License
[![MorphoSkel3D Code License: BSD-2-Clause](https://img.shields.io/badge/MorphoSkel3D%20Code%20License-BSD--2--Clause-blue.svg)](LICENSE)

## Citation
If you find *MorphoSkel3D* useful to your research, please consider citing:
```bibtex
@INPROCEEDINGS{11125643,
  author={Onghena, Pierre and Velasco-Forero, Santiago and Marcotegui, Beatriz},
  booktitle={2025 International Conference on 3D Vision (3DV)}, 
  title={MorphoSkel3D: Morphological Skeletonization of 3D Point Clouds for Informed Sampling in Object Classification and Retrieval}, 
  year={2025},
  volume={},
  number={},
  pages={1350-1359},
  keywords={Point cloud compression;Geometry;Training;Solid modeling;Three-dimensional displays;Shape;Surface morphology;Sampling methods;Skeleton;Surface treatment},
  doi={10.1109/3DV66043.2025.00128}
}
```
