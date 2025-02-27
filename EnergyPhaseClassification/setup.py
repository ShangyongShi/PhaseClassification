#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 15:18:10 2025

@author: sshi
"""
from setuptools import setup, find_packages

setup(
    name="EnergyPhase",
    version="2.0",
    description="Precipitation Phase Partitioning based on Atmospheric Energy, Shi and Liu (2024)",
    author="Shangyong Shi",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "xarray",
        "joblib"
    ],
    python_requires=">=3.7",
)
