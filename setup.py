#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="bayesopt_openmdao",
    version="0.1",
    packages=find_packages(exclude=["examples"]),

    install_requires=["openmdao>=1.7.1", "bayesopt"]
)
