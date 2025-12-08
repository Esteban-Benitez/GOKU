# GOKU - Deep Generative ODE Modelling with Known Unknowns

This repository is an implementation of the GOKU paper: [Generative ODE Modeling with Known Unknowns](https://arxiv.org/abs/2003.10775).

### Installation

Install Astral UV

```sh

curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
uv sync

```


### Data creation
To create the datasets used in the paper run:
* Friction-less pendulum:  `uv run python create_data.py --model pendulum`
* Friction pendulum: `uv run python create_data.py --model pendulum --friction`
* Double-pendulum experiment:  `uv run python create_data.py --model double_pendulum`
* Cardiovascular system: `uv run python create_data.py --model cvs`

The data would be created using default arguments. To view / modify them check the file `config.py`, and `create_data.py`.

### Training
To train the GOKU model run: `uv run goku_train.py --model <pendulum/pendulum_friction/double_pendulum/cvs>`

To train baselines:

* Latent-ODE: `uv run python latent_ode_train.py --model <pendulum/pendulum_friction/double_pendulum/cvs>`.
* LSTM: `uv run python lstm_train.py --model <pendulum/pendulum_friction/double_pendulum/cvs>`.
* Direct-Identification (DI) has 3 different files for the different datasets (it cannot run the friction pendulum, since it needs the entire ODE functional form):
  * Pendulum: `uv run python di_baseline_pendulum.py`
  * Double Pendulum: `uv run python di_baseline_double_pendulum.py`
  * CVS: `uv run python di_baseline_cvs.py`

```zsh
uv run python goku_train.py --model pendulum
uv run python latent_ode_train.py --model pendulum
```
  
### Requirements:
* python 3
* pytorch
* numpy
* gym (for the pendulum and double pendulum experiments)

### Evaluation Scripts

```sh

./evaluate/adam/script.sh
./evaluate/adamw/script.sh
uv run python plot_eval.py
```