#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name='tgym',
    version='0.1.15',
    description="Trading Gym is an open-source project for the development of reinforcement learning algorithms in the context of trading.",
    packages=find_packages(),
    install_requires=[
        'matplotlib==2.0.2',
        'keras >= 2.0.8',
        'tensorflow'
    ],
    license="MIT license",
    zip_safe=False,
    keywords='tgym'
)
