# Trust, but verify

Anonymized code for "Trust, but verify: model-based exploration in sparse reward environments" https://openreview.net/forum?id=DE0MSwKv32y

Tested on Ubuntu 18.04, Python 3.6
## Quickstart

### Install

`pip install -e .[dev]`

The `-e` option allows you to make changes to the package that will be immediately visible in the virtualenv.

### Run

The entrypoint for running experiments resides in `alpacka/runner.py`. Different experiment settings are defined via [Gin](https://github.com/google/gin-config) config files. 
All configs can be found in `configs` directory. To run for example BestFS agent on the Tower of Hanoi environment with `TBV` mechanism use `configs/hanoi/bfs_hanoi_tbv.gin` config:
```
python3 -m alpacka.runner \
    --config_file=configs/hanoi/bfs_hanoi_tbv.gin \
    --output_dir=./out
```

For machines with small RAM this might end up with memory error. If you just 
want to follow code execution you can run quick experiment with down-scaled settings:

```
python3 -m alpacka.runner \
    --config_file=configs/hanoi/bfs_hanoi_tbv.gin \
    --output_dir=./out
    --config=Runner.episode_time_limit=50
    --config=Runner.batch_stepper_class=@alpacka.batch_steppers.LocalBatchStepper
    --config=Runner.n_envs=1
    --output_dir=./out
    --config=Runner.n_precollect_epochs=1
    --config=Runner.n_model_precollect_episodes=100
    --config=Runner.n_model_pretrain_epochs=1
```
## Design overview

### Runner

[`Runner`](alpacka/runner.py) is the main class of the experiment, taking care of running the training loop. Each iteration consists of two phases: gathering data from environments and training networks.

### Agent & Trainer

The logic of an RL algorithm is split into two classes: [`Agent`](alpacka/agents/base.py) collecting experience by trying to solve the environment and [`Trainer`](alpacka/trainers/base.py) training the neural networks on the collected data.

### Network

[`Network/TrainableNetwork`](alpacka/networks/core.py) abstracts out the deep learning framework used for network inference and training.

### BatchStepper

[`BatchStepper`](alpacka/batch_steppers.py) is an abstraction for the parallelization strategy of data collection. It takes care of running a batch of episodes with a given `Agent` on a given `Env` using a given `Network` for inference. We currently support 2 strategies:

- Local execution - `LocalBatchStepper`: Single node, `Network` inference batched across `Agent` instances. Running the environments is sequential. This setup is good for running light `Env`s with heavy `Network`s that benefit a lot from hardware acceleration, on a single node with or without GPU.
- Distributed execution using [Ray](https://ray.readthedocs.io/en/latest/) - `RayBatchStepper`: Multiple workers on single or multiple nodes, every worker has its own `Agent`, `Env` and `Network`. This setup is good for running heavy `Env`s and scales well with the number of nodes.
