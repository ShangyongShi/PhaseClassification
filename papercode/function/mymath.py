#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 11:35:28 2023

@author: sshi
"""
import numpy as np
# linear function every two points 
def linear(t1, p1, t2, p2):
    slope = (p2 - p1)/(t2 - t1)
    intercept = p1 - slope*t1
    return slope, intercept

