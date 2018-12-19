# Trading Gym

This project is a modified version of the original [Trading Gym](https://github.com/thedimlebowski/Trading-Gym), which is an open-source project for the development of reinforcement learning algorithms in the context of trading.
It is currently composed of a single environment and implements a generic way of feeding this trading environment different type of price data.

## Demo
*Perfect* Sine-Function Trading
![](https://raw.githubusercontent.com/workofart/RL-trading/master/Perfect.gif)

*Semi-Perfect* Sine-Function Trading
![](https://raw.githubusercontent.com/workofart/RL-trading/master/SemiPerfect.gif)


## Installation

1. Make sure you have **[Python 3](https://www.python.org/downloads/)**

2. Then install trading-gym

    ```shell
    pip install tgym
    ```

    We strongly recommend using virtual environments. A very good guide can be found at https://python-guide-pt-br.readthedocs.io/pt_BR/latest/dev/virtualenvs.html.

3. Then install keras
    ```shell
    pip install keras
    ```
    By default, the backend will be set to Theano. You can also run it with Tensorflow by installing it with `pip install tensorflow`. You then need to edit `~/.keras/keras.json` and make sure `"backend": "tensorflow"` is specified.

## Running

```shell
python examples/dqn_agent.py
```

## Configurations
Stored in `config/config.json`

## Notes
To plot properly with `matlibplot`, you should disable "interactive mode" by adding this in your python scripts (I've added it to `examples/dqn_agent.py` already)
```python
plt.interactive(False)
```

## Compatibility with OpenAI gym

Our environments API is strongly inspired by [OpenAI](https://gym.openai.com/) Gym. We aim to entirely base it upon OpenAI Gym architecture and propose Trading Gym as an additional OpenAI environment.

## License

[MIT License](https://github.com/workofart/RL-trading/blob/master/LICENSE)
