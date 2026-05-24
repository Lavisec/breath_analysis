#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 24 09:56:47 2026

@author: aviv
"""

from scipy.signal import butter, filtfilt, find_peaks
from scipy.optimize import curve_fit
from scipy.cluster.vq import kmeans, vq
import numpy as np

pressure = results['pressure_resampled_lp']
samp_rate = 1000
t = results['time_resampled']
DC_EXCLUSION_FREQ = 0.04
# t_100Hz = np.arange(t[0], t[-1], 1/samp_rate)
# old_bl = np.interp(t_100Hz, t, results['baseline'])

inhale_amp_th = results['inhale_parameters']['threshold_dict']['amplitude_threshold']
exhale_amp_th = results['exhale_parameters']['threshold_dict']['amplitude_threshold']

# pressure_100Hz = np.interp(t_100Hz, t, pressure)
# low = 0.1
# high = 3

# filtered = filtfilt(*butter(3, [low, high], btype='bandstop', fs=samp_rate), pressure_100Hz)

# hist, bins = np.histogram(filtered, bins=1000, range=(-0.2, 0.2))
# bins = (bins[:-1] + bins[1:])/2
# max_ind = np.argmax(hist)
# baseline = bins[max_ind]

# popt, _ = curve_fit(DC_EXCLUSION_FREQ = 0.04
#     lambda x, a, b, c: a*np.exp(-(x-b)**2/(2*c**2)),
#     bins,
#     hist,
#     p0=[hist.max(), bins[np.argmax(hist)], 1]
# )

# n = len(pressure)
# fft_vals = np.abs(np.fft.rfft(pressure))
# fft_freqs = np.fft.rfftfreq(n, d=1.0 / samp_rate)

# fft_vals[fft_freqs < DC_EXCLUSION_FREQ] = 0
# dominant_freq = fft_freqs[np.argmax(fft_vals)]

# duration_th = 0.5 * samp_rate/dominant_freq

# peaks_aux, props = find_peaks(pressure, distance=duration_th, height=0)
# peaks_vals = pressure[peaks_aux]

# k=3
# centroidpeaks, props = find_peaks(pressure, distance=duration_th, height=amp_th, prominence=amp_th)
# peaks_vals = pressure[peaks], distortion = kmeans(peaks_vals, k)
# labels, distances = vq(peaks_vals, np.sort(centroids))

# amp_th = np.min(peaks_vals[labels == 1]) / 2
# amp_th = results['inhale_parameters']['threshold_dict']['amplitude_threshold']
# amp_th=filtered

# peaks, props = find_peaks(pressure, distance=duration_th, height=amp_th, prominence=amp_th)
# peaks_vals = pressure[peaks]



peak_ind = results['inhale_parameters']['parameters']['peaks'][:, 0].astype(int)

plt.figure(1)
# plt.plot(t_100Hz, pressure_100Hz)
# plt.axhline(baseline, color='orange')
# plt.plot(t_100Hz, old_bl, color='black')
# plt.axhline(popt[1], color='magenta')
# plt.plot(t_100Hz, filtered, color='green')
# plt.plot(results['pressure_bc'])
plt.plot(t, pressure)
plt.axhline(inhale_amp_th, linestyle='--', color='gray')
plt.axhline(exhale_amp_th, linestyle='--', color='gray')
# plt.plot(t[peaks], peaks_vals, linestyle='none', marker='o')
plt.plot(t[peak_ind], results['inhale_parameters']['parameters']['peaks'][:, 1], 
         linestyle='none', marker='x', color='black')
plt.grid()


# plt.figure(2)
# plt.plot(bins, hist)
# plt.plot(
#     bins,
#     popt[0]*np.exp(-(bins-popt[1])**2/(2*popt[2]**2)),
#     label='gaussian fit'
# )
# plt.axvline(popt[1])


