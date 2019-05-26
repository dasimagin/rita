# Rita (Reinforced in team agent)
Reinforcement Learning project by HSE students: [Victor Kukanov](https://github.com/granilace), [Roman Sokolov](https://github.com/sokolov5roma), [Maxim Kobelev](https://github.com/flyinslowly), [Daniil Gontar](https://github.com/Danyago98). Mentor: [Denis Simagin](https://github.com/dasimagin).


## About
This is an PyTorch implementation of [Asynchronous Advantage Actor-Critic](https://arxiv.org/pdf/1602.01783) with intrinsic [curiosity-based](https://pathak22.github.io/noreward-rl/resources/icml17.pdf) reward. 

Tested on [DeepMind Lab](https://github.com/deepmind/lab) and gym-like environments.

## Dependencies
* Python 3.6
* PyTorch, gym, NumPy 
* [Bazel](https://docs.bazel.build/versions/master/install.html)
* DeepMind Lab

## How to run

### Train
The following command will train an A3C agent:
```bash
python train.py --experiment-folder *path_to_config*
```
For example this command should run training process in DeepMind Lab deathmatch environment with 4 bots:
```bash
python train.py --experiment-folder experiments/Dmlab_LtHorseshoe
```

There are some ready-to-run configs in ``experiments`` folder:
* ``Atari_SpaceInvaders`` – [gym](https://gym.openai.com/envs/SpaceInvaders-v0) env of Atari SpaceInvaders
* ``Dmlab_SeekAvoid`` – simple DeepMind Lab arena
* ``Dmlab_SmallStaticMaze`` – small static DeepMind Lab maze
* ``Dmlab_SmallRandomMaze`` - small random DeepMind Lab maze
* ``Dmlab_LtHorseshoe`` – deatchmatch DeepMind Lab map with 4 easy bots

### Evaluate
You can record a video of gameplay of your trained agent:
```bash
python evaluate.py --experiment-folder *config_path* --pretrained-weights *weights_path*
```
