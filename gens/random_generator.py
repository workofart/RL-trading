"""
In this example we show how a random generator is coded.
All generators inherit from the DataGenerator class
The class yields tuple (bid_price,ask_price)
spread = ask_price - bid_price
limit range between range_low, range_high
better format for return value
"""
from __future__ import absolute_import 
import numpy as np
from tgym.core import DataGenerator


class RandomGenerator(DataGenerator):

    @staticmethod
    def _generator(spread=0.001, range_low=1.0, range_high=2.0, round_len=4):
        while True:
            # val = np.random.randn()
            val = np.random.uniform(low=range_low, high=range_high)
            ask_price, bid_price = round(val, round_len), round(val + spread, round_len)
            yield ask_price, bid_price