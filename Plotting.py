#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 09:45:40 2026

@author: aviv
"""


from scipy.signal import decimate

# Plot data with inhale se points

plt.plot(result['pressure_resampled'], label='Resampled', color='blue', alpha=0.6, linewidth=1)
plt.plot(result['inhale_start_end_points'][:, 0], result['pressure_resampled'][result['inhale_start_end_points'][:, 0]],
         color='red', alpha=0.6, linewidth=1, marker='o', linestyle='None', label='Inhale Start')
plt.plot(result['inhale_start_end_points'][:, 1], result['pressure_resampled'][result['inhale_start_end_points'][:, 1]],
         color='green', alpha=0.6, linewidth=1, marker='o', linestyle='None', label='Inhale End')


#%%

# Plot all peaks
sorted_peak = np.sort(peak.flatten())
aux = np.diff(np.log(sorted_peak))

z_aux = (aux - np.mean(aux)) / np.std(aux)



# plt.plot(sorted_peak, linestyle='None', marker='o', color='blue')
plt.plot(np.log(sorted_peak), linestyle='None', marker='o', color='red')
plt.plot(aux*100, linestyle='None', marker='o', color='green')
plt.plot(z_aux, linestyle='None', marker='o', color='blue')
     
# plt.hist(aux, bins=1000)

#%%

# Plot data with peaks outliers

# plt.plot(result['pressure_resampled'], label='Resampled', color='blue', alpha=0.6, linewidth=1)
# plt.plot(result['pressure_resampled'][delete_points_peaks], result['parameters']['peak'][delete_points_peaks],
#          color='red', alpha=0.6, linewidth=1, marker='o', linestyle='None')


plt.plot(result['pressure_resampled'])
plt.plot(result['parameters']['peak'][:, 0], result['parameters']['peak'][:, 1], color='red', linestyle='None', marker='o')
bad_ind = peaks.loc[peaks['mask'] == 0, 'idx'].astype(int)
plt.plot(bad_ind, result['pressure_resampled'][bad_ind] + 0.1, color='black', linestyle='None', marker='o')


#%%

#Plot all peaks and troughs

plt.plot(results['pressure_resampled'])
plt.plot(results['inhale_parameters']['parameters']['peaks'][:, 0],
                  results['inhale_parameters']['parameters']['peaks'][:, 1],
                  color='red', linestyle='None', marker='o')
plt.plot(results['exhale_parameters']['parameters']['troughs'][:, 0],
                  results['exhale_parameters']['parameters']['troughs'][:, 1],
                  color='green', linestyle='None', marker='o')
plt.axhline(y=results['inhale_parameters']['threshold_dict']['amplitude_threshold'], color='b', linestyle='--')
plt.axhline(y=results['exhale_parameters']['threshold_dict']['amplitude_threshold'], color='b', linestyle='--')

#%%

for peak in breaths:
    plt.plot(peak)
    
#%%

plt.plot(result['pressure_filtered'], label='data')
plt.plot(baselines[0], label='1 window')
plt.plot(baselines[1], label='2 window')
plt.plot(baselines[2], label='4 window')
plt.plot(baselines[3], label='8 window')
plt.plot(baselines[4], label='16 window')
plt.plot(baseline, label='final baseline')
plt.legend()

#%%
target_diffs
plt.plot(result['pressure_resampled'])
# above_amp_ind = np.where(results['pressure_resampled'] > results['inhale_parameters']['threshold_dict']['amplitude_threshold']) 
# plt.plot(above_amp_ind[0], results['pressure_resampled'][above_amp_ind], marker='o', linestyle='None')
plt.plot(result['inhale_parameters']['se_points'][:, 0],
         result['pressure_resampled'][result['inhale_parameters']['se_points'][:, 0]], marker='o', color='black', linestyle='None')
plt.plot(result['inhale_parameters']['se_points'][:, 1],
         result['pressure_resampled'][result['inhale_parameters']['se_points'][:, 1]], marker='x', color='black', linestyle='None')


#%%

plt.plot(result['pressure_resampled'], color='red')
plt.plot(delete_points_peaks, result['pressure_resampled'][delete_points_peaks], linestyle='None', color='black', 
         marker='o')

#%%
plt.figure()
plt.plot(results['pressure_resampled'])
for event in results['event_list']:
    if event['type'] == 'inhale':
        plt.plot(np.arange(event['start'], event['end']), results['pressure_resampled'][event['start']:event['end']], color='red')
        plt.plot(event['extrimum'][0], results['pressure_resampled'][event['extrimum'][0].astype(int)], linestyle='None', marker='o', color='orange', mfc='none')
    elif event['type'] == 'exhale':
        plt.plot(np.arange(event['start'], event['end']), results['pressure_resampled'][event['start']:event['end']], color='green')
        plt.plot(event['extrimum'][0], results['pressure_resampled'][event['extrimum'][0].astype(int)], linestyle='None', marker='o', color='orange', mfc='none')
    else:
        plt.plot(np.arange(event['start'], event['end']), results['pressure_resampled'][event['start']:event['end']], color='black', linestyle='--')
    
    plt.plot(event['start'], results['pressure_resampled'][event['start']], linestyle='None', marker='o', color='magenta', mfc='none')
    
plt.axhline(results['inhale_parameters']['threshold_dict']['amplitude_threshold'], linestyle='--', color='cyan')
plt.axhline(results['exhale_parameters']['threshold_dict']['amplitude_threshold'], linestyle='--', color='cyan')
        
#%%

plt.plot(result['pressure_resampled'])
plt.plot(se_points[:, 0],
         result['pressure_resampled'][se_points[:, 0]], marker='o', color='black', linestyle='None')
plt.plot(se_points,
         result['pressure_resampled'][se_points[:, 1]], marker='x', color='black', linestyle='None')

plt.axhline(amplitude_threshold, color='orange', linestyle='--')

#%%

plt.plot(results['pressure_resampled'])
plt.plot(results['inhale_parameters'])

#%%


pressure = results[0]['pressure_raw']
N=len(pressure) // 2
fft_vals = np.fft.fft(pressure)
fft_freqs = np.fft.fftfreq(len(pressure), d=min_dt)

new_fft = np.log(np.abs(fft_vals[:N]))
# plt.plot(fft_freqs[:N],new_fft , linestyle='none', marker='o', mfc='none')

x = np.arange(0,20,0.02)

vals_interp = decimate(new_fft, q=N//len(x))


plt.plot(x, vals_interp[:1000])

#%%

plt.plot(data['pressure_upsampled'])

for event in data['event_list']:
    if event['type'] == 'inhale':
        plt.plot(np.arange(event['start'], event['end']), data['pressure_upsampled'][event['start']:event['end']], color='red')
        # plt.plot(event['extrimum'][0], results['pressure_resampled'][event['extrimum'][0].astype(int)], linestyle='None', marker='o', color='orange', mfc='none')
    elif event['type'] == 'exhale':
        plt.plot(np.arange(event['start'], event['end']), data['pressure_upsampled'][event['start']:event['end']], color='green')
        # plt.plot(event['extrimum'][0], results['pressure_resampled'][event['extrimum'][0].astype(int)], linestyle='None', marker='o', color='orange', mfc='none')
    else:
        plt.plot(np.arange(event['start'], event['end']), data['pressure_upsampled'][event['start']:event['end']], color='black', linestyle='--')
    
    plt.plot(event['start'], data['pressure_upsampled'][event['start']], linestyle='None', marker='o', color='magenta', mfc='none')
    
plt.plot(data['inh_amp_th'], linestyle='--', color='cyan')
plt.plot(data['inh_amp_th'], linestyle='--', color='cyan')

#%% 

ticks_font = 14
labels_font = 14
legend_font = 14

plt.xticks(fontsize=ticks_font)
plt.yticks(fontsize=ticks_font)

plt.plot(dataset['time'], dataset['pressure'], label='Raw Data')
plt.plot(dataset['time'], dataset['inh_amp_th'], label='Inhale Amplitude Threshold', linestyle='--', color='cyan')
plt.plot(dataset['time'], dataset['exh_amp_th'], label='Inhale Amplitude Threshold', linestyle='--', color='magenta')
plt.xlabel('Time [s]', fontsize=labels_font)
plt.ylabel('Pressure [A.U.]', fontsize=labels_font)

# plt.axvline(data['time_raw'][0], linestyle='--', color='red', label='First Window')
# plt.axvline(data['time_raw'][(len(data['time_raw']) // 2) - 50], linestyle='--', color='red')

# plt.axvline(data['time_raw'][len(data['time_raw']) // 4], linestyle='--', color='black', label='Second Window')
# plt.axvline(data['time_raw'][(3 * len(data['time_raw'])) // 4], linestyle='--', color='black')

# plt.axvline(data['time_raw'][(len(data['time_raw']) // 2) + 50], linestyle='--', color='cyan', label='Third Window')
# plt.axvline(data['time_raw'][len(data['time_raw'])-1], linestyle='--', color='cyan')


plt.legend(fontsize=legend_font)
plt.grid()
























