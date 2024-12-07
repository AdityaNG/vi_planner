# vi_planner

[![codecov](https://codecov.io/gh/AdityaNG/vi_planner/branch/main/graph/badge.svg?token=vi_planner_token_here)](https://codecov.io/gh/AdityaNG/vi_planner)
[![CI](https://github.com/AdityaNG/vi_planner/actions/workflows/main.yml/badge.svg)](https://github.com/AdityaNG/vi_planner/actions/workflows/main.yml)
[![GitHub License](https://img.shields.io/github/license/AdityaNG/vi_planner)](https://github.com/AdityaNG/vi_planner/blob/main/LICENSE)
[![PyPI - Version](https://img.shields.io/pypi/v/vi_planner)](https://pypi.org/project/vi_planner/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/vi_planner)


This repo vi_planner is a pip module of the ViPlanner paper.
ViPlanner is a robust learning-based local path planner based on semantic and depth images.
Fully trained in simulation, the planner can be applied in dynamic indoor as well outdoor environments.
We provide it as an extension for [NVIDIA Isaac-Sim](https://developer.nvidia.com/isaac-sim) within the [IsaacLab](https://isaac-sim.github.io/IsaacLab/) project (details [here](./omniverse/README.md)).
Furthermore, a ready to use [ROS Noetic](http://wiki.ros.org/noetic) package is available within this repo for direct integration on any robot (tested and developed on ANYmal C and D).

**Keywords:** Visual Navigation, Local Planning, Imperative Learning

## <a name="CitingViPlanner"></a>Citing ViPlanner
The model is from the ViPlanner paper
```
@article{roth2023viplanner,
  title     ={ViPlanner: Visual Semantic Imperative Learning for Local Navigation},
  author    ={Pascal Roth and Julian Nubert and Fan Yang and Mayank Mittal and Marco Hutter},
  journal   = {2024 IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2023},
  month     = {May},
}
```

## <a name="CitingSOccDPT"></a>Citing SOccDPT

The plotting utility is from the SOccDPT paper
```
@article{
  nalgunda2024soccdpt, 
  author = {Aditya Nalgunda Ganesh},
  title = {SOccDPT: 3D Semantic Occupancy from Dense Prediction Transformers trained under memory constraints},
  journal = {Advances in Artificial Intelligence and Machine Learning},
  volume = {4}, 
  number = {2},
  pages = {2201--2212},
  year = {2024}
}
```


## Install it from PyPI

```bash
pip install vi_planner
```

## Usage

```py
from vi_planner import VIPlanner
import cv2

planner = VIPlanner()
goal = [5, 0, 0]
frame = cv2.imread("image.png")

model_output = planner.run(frame, [5, 0, 0])

vis_frame = planner.visualize(frame, model_output)

cv2.imwrite("output.png", vis_frame)
```

You can run the CLI demo
```bash
$ python -m vi_planner
#or
$ vi_planner

usage: vi_planner [-h] [--headless | --no-headless] [--save SAVE] [--goal GOAL GOAL GOAL] video_path

VI-Planner demo

positional arguments:
  video_path            Path to input video file

options:
  -h, --help            show this help message and exit
  --headless, --no-headless
                        No display mode
  --save SAVE           Path to save output video
  --goal GOAL GOAL GOAL
                        Goal coordinates (x,y,z)
```

## Development

Read the [CONTRIBUTING.md](CONTRIBUTING.md) file.
