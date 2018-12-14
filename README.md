# Trading Gym

This project is a modified version of the original [Trading Gym](https://github.com/thedimlebowski/Trading-Gym), which is an open-source project for the development of reinforcement learning algorithms in the context of trading.
It is currently composed of a single environment and implements a generic way of feeding this trading environment different type of price data.

## Installation

`pip install tgym`

We strongly recommend using virtual environments. A very good guide can be found at http://python-guide-pt-br.readthedocs.io/en/latest/dev/virtualenvs/.


![]()

## Create your own `DataGenerator`


## Compatibility with OpenAI gym

Our environments API is strongly inspired by OpenAI Gym. We aim to entirely base it upon OpenAI Gym architecture and propose Trading Gym as an additional OpenAI environment.

## Examples

To run the `dqn_agent.py` example, you will need to also install keras with `pip install keras`. By default, the backend will be set to Theano. You can also run it with Tensorflow by installing it with `pip install tensorflow`. You then need to edit `~/.keras/keras.json` and make sure `"backend": "tensorflow"` is specified.

## License

MIT License