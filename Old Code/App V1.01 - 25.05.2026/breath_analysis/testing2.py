#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 16:36:17 2026

@author: aviv
"""

from scipy.signal import butter, filtfilt, find_peaks, hilbert
from scipy.optimize import curve_fit
from scipy.cluster.vq import kmeans, vq
import numpy as np

pressure = results['pressure_bc']
samp_rate = 100
t = results['time']
DC_EXCLUSION_FREQ = 0.04

rect_pressure = pressure
rect_pressure[rect_pressure < 0] = 0

filtered = filtfilt(*butter(4, 0.01, btype='lowpass', fs=samp_rate), rect_pressure)

plt.plot(pressure)
plt.plot(filtered)

