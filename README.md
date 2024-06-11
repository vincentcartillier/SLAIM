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


Code tested on a single A40 node (Ubundu 20, cuda 11.3) <br />

## Setup
1. Clone SLAIM
```bash
git clone --recursive https://github.com/vincentcartillier/SLAIM
cd SLAIM
# git submodule update --init --recursive
```

2. Create a conda environment,
```bash
conda create -n slaim python=3.7
conda activate slaim
pip install -r requirements.txt
```

3. Install Instant-NGP.  <br />
You will need GCC/G++ 8 or higher, cmake v3.21 or higher and cuda 10.2 or higher. Please check the [main repo](https://github.com/NVlabs/instant-ngp) for more details.


```bash
cd dependencies/instant-ngp
cmake -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DNGP_BUILD_WITH_GUI=off . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo -j
```

Note: I had to manully link GLEW in the CMakeLists.txt (L180). If you already have GLEW installed this shouldn't be needed. Check the CMakeLists.txt from main repo for reference.

```
[...(L180)]
if (MSVC)
	list(APPEND NGP_INCLUDE_DIRECTORIES "dependencies/gl3w")
	list(APPEND GUI_SOURCES "dependencies/gl3w/GL/gl3w.c")
	list(APPEND NGP_LIBRARIES opengl32 $<TARGET_OBJECTS:glfw_objects>)
else()
	find_package(GLEW REQUIRED)
    set(GLEW_INCLUDE_DIRS "/nethome/vcartillier3/lib/glew-2.1.0/build/cmake/install/include")
    set(GLEW_LIBRARIES "/nethome/vcartillier3/lib/glew-2.1.0/build/cmake/install/lib/libGLEW.so")
	list(APPEND NGP_INCLUDE_DIRECTORIES ${GLEW_INCLUDE_DIRS})
	list(APPEND NGP_LIBRARIES GL ${GLEW_LIBRARIES} $<TARGET_OBJECTS:glfw_objects>)
endif()
```

## Data

### Download
* **ScanNet:** <br />
Please follow the data downloading procedure on [ScanNet](http://www.scan-net.org/) website, and extract color/depth frames from the `.sens` file using this [code](https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/reader.py). <br />
Place the data under `./Datasets/scannet/scans/scene0000_00/frames`

* **Replica:** <br />
Use the following script to download the data. Data is saved under `./Datasets/Replica`. We use the same trajectories and scenes as in iMAP, NICE-SLAM, Co-SLAM etc...
```bash
bash scripts/download_replica.sh
```

* **TUM RGB-D:** <br />
Use the following script to download the data. Data is saved under `./Datasets/TUM-RGBD`.
```bash
bash scripts/download_tum.sh
```

### Pre-process
The following steps show how to pre-process the data for a given scene. <br />
All of the following commands are compiled in the following bash script:
```bash
bash tools_experiments/make_data.sh configs/Replica/replica_office0.yml
```


1. preprocess camera poses:
```
python tools_make_data/preprocess_camera_poses.py --config configs/Replica/replica_office0.yml
```
This will create an experiment folder under `data/` with the updated config file and preprocessed camera poses.

2. Build point cloud (for scene scaling):
```
python tools_make_data/save_input_pc_for_init_scene_scaling.py --config data/Replica_office0/0/configs.yml
```
NOTE: if you turn off the `POSES_FILENAME` variable (ie "") in config file, that script will save the raw PC without using preprocessed poses.

3. Estimate scaling parameters:
```
python tools_make_data/estimate_scale_and_shift_using_GT_pc.py --config data/Replica_office0/0/configs.yml
```

4. Prepare data for NGP:
```
python tools_make_data/prepare_ngp_format_dataset.py --config data/Replica_office0/0/configs.yml
python tools_make_data/add_poses_to_transforms_json.py --config data/Replica_office0/0/configs.yml
```

## RUN
Replica <output_dir> with the target directory (eg. `./outputs/`).

* **ScanNet:** <br />
```bash
bash tools_experiments/run.sh <output_dir> data/ScanNet_scene0000/0/configs.yml configs/ScanNet/experiment_hyperparams/base.yml
```

* **Replica:** <br />
```bash
bash tools_experiments/run.sh <output_dir> data/Replica_office0/0/configs.yml configs/Replica/experiment_hyperparams/base.yml
```
If you wish to do 3D reconstruction evaluation, you will need to download the corresponding data for mesh culling on the [neural_slam_eval](https://github.com/JingwenWang95/neural_slam_eval) repo and place it under `dependencies/neural_slam_eval/data/CoSLAM_data`. And you'll also have to symlink the Replica data to that repo
```
cd dependencies/neural_slam_eval/data
ln -s ../../../Datasets/Replica/
cd ../../../Datasets/
ln -s ../dependencies/neural_slam_eval/data/CoSLAM_data/
```
You'll also need to setup the corresponding conda env. Note that you may want to manually compile Open3d with headless rendering if you are working on a cluster for instance. Please refere to this [page](https://www.open3d.org/docs/latest/tutorial/Advanced/headless_rendering.html) for instruction on how to compile.
```bash
conda create -n slaim_3d_eval python=3.8
conda activate slaim_3d_eval
pip install -r dependencies/neural_slam_eval/requirements.txt
```
Then run:
```bash
bash tools_experiments/run_replica_eval_3D.sh <output_dir> data/Replica_office0/0/configs.yml configs/Replica/experiment_hyperparams/base.yml
```

* **TUM RGB-D:** <br />
```bash
bash tools_experiments/run.sh <output_dir> data/TUM_desk/0/configs.yml configs/TUM/experiment_hyperparams/base.yml
```



## Acknowledgment
Our code is based of the original Instant-NGP [repo](https://github.com/NVlabs/instant-ngp). We thank @Thomas Muller (a.k.a Tom94) for their help answering our questions while developping. We also thank the folks from [Co-SLAM](https://github.com/HengyiWang/Co-SLAM) from sharing their code early to us. Thanks a lot for the awesome repos and making code available to the community!

