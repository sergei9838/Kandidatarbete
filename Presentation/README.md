# Presentation med Manim

## Installation
```
conda create -n "manim_env" python=3.10.8
conda activate manim_env
conda install -c conda-forge manim
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

## Running
To run code use with low quality (fast, good for testing) use
```
manim -pql KDEScene.py KDEScene
```
and for high quality (slow) use
```
manim -pqh KDEScene.py KDEScene
```