# <ins>T</ins>ext-<ins>A</ins>ware <ins>D</ins>iffusion for <ins>Po</ins>licy <ins>Le</ins>arning (TADPoLe) :frog:
## Quick Start
### Setup Repository and Environment
```shell
git clone https://github.com/brown-palm/tadpole.git
cd tadpole

conda create -n tadpole python=3.9
conda activate tadpole

pip install -r requirement.txt
```
### Customize Configurations for Dog and Humanoid Environments
- Replace the following files in `<conda path>/envs/tadpole/lib/python3.9/site-packages/dm_control/suite/` with corresponding ones in `custom_envs/`:

    | Source File in `dm_control/suite/` | Customized File in `custom_envs/` |
    |-----------------------------------------------------|-----------------------------------|
    | `dog.py`                                            | `dog/dog.py`                      |
    | `dog.xml`                                           | `dog/dog.xml`                     |
    | `humanoid.py`                                       | `humanoid/humanoid.py`            |
    | `humanoid.xml`                                      | `humanoid/humanoid.xml`           |

- We provide background image files `skybox.png` and `grass.png` in `custom_envs/common_assets/`. In `dog.xml` and `humanoid.xml`, input the correct path for background images at lines below: 
    - `<texture type="skybox" file="[path to skybox.png]" gridsize="3 4" gridlayout=".U..LFRB.D.." width="8192" height="8192"/>`
    - `<texture name="grass" file="[path to grass.png]" type="2d"/>`

## Policy Learning
>To enable wandb logging for experiments, fill in the `wandb_entity` in `cfgs/default.yaml` and add `use_wandb=True` to the commands below.
### Goal Achievement with TADPoLe
Humanoid Stand:
```shell
python src/train_dmc.py \
    task="humanoid-stand" \
    text_prompt="a person standing" \
    seed=0 \
    noise_level_base=400 
```

Dog Stand:
```shell
python src/train_dmc.py \
    task="dog-stand" \
    text_prompt="a dog standing" \
    seed=0 \
    noise_level_base=400 
```

Humanoid Novel Goal Achievement:
```shell
python src/train_dmc.py \
    task="humanoid-stand" \
    text_prompt="a person doing splits" \
    seed=0 \
    noise_level_base=400 
```


### Robotic Manipulation with TADPoLe
MetaWorld task examples: `"metaworld-door-close"`, `"metaworld-window-open"`, `"metaworld-coffee-push"`, etc.

```shell
python src/train_mw.py \
    task="metaworld-door-close" \
    text_prompt="closing a door" \
    seed=0 \
    noise_level_base=500 
```

### Continuous Locomotion with Video-TADPoLe
Humanoid Walk:
```shell
python src/train_dmc.py \
    task="humanoid-walk" \
    text_prompt="a person walking" \
    seed=0 \
    tadpole_type="video-tadpole" \
    noise_level_base=500 
```

Dog Walk:
```shell
python src/train_dmc.py \
    task="dog-walk" \
    text_prompt="a dog walking" \
    seed=0 \
    tadpole_type="video-tadpole" \
    noise_level_base=500 
```

## Citation
If you find this repository helpful, please consider citing our work:
```bibtex
@inproceedings{luo2024text,
  author={Luo, Calvin and He, Mandy and Zeng, Zilai and Sun, Chen},
  booktitle={Advances in Neural Information Processing Systems},
  title={Text-Aware Diffusion for Policy Learning},
  volume={37},
  year={2024}
}
```

## Acknowledgement
This repo contains code adapted from [tdmpc](https://github.com/nicklashansen/tdmpc). We thank the authors and contributors for open-sourcing their code.
