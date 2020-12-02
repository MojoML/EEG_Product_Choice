from scipy.signal import butter, lfilter
from scipy import signal
import mne
from scipy.integrate import simps
from scipy import signal
import pandas as pd
import seaborn as sns
from scipy.fft import rfftn
from scipy.fft import rfftfreq
from scipy.fft import fftn
from scipy.fft import fftfreq
import pywt
import itertools
import numpy as np



def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='bandpass')
    y = lfilter(b, a, data, axis = 1)
    return y


def spectral_centroid(x, samplerate=128):
    magnitudes = np.abs(np.fft.rfft(x, axis =2))
    length = len(x)
    freqs = np.abs(np.fft.fftfreq(length, 1.0/samplerate)[:length//2+1])
    magnitudes = magnitudes[:length//2+1]
    return np.sum(magnitudes*freqs) / np.sum(magnitudes) 





# filters data from pandas dataframe, initialize sample frequency, low_cut freq, high cut freq, nc, nt
def filter_data(data: pd.DataFrame, sfreq, lfreq, hfreq, nc, nt, names):
    filtered_x = mne.filter.filter_data(data.values.reshape(1045, nc, nt), sfreq= sfreq, h_freq = hfreq, l_freq = lfreq)
    filtered_x_df = pd.DataFrame(filtered_x.reshape(data.shape[0], -1), index=names, columns=np.arange(nc*nt), dtype='float64')                               
    
    return filtered_x_df

# visualize signal
def visualize_signal(data, sfreq,tmp_signal):
    if isinstance(data, pd.DataFrame) and data.values.shape != (1045,14,512):
        data = data.values.reshape(1045,14,512)
    ch_types = ["eeg"] * 14 
    ch_names = ['AF3', 'F7', 'F3', 'FC5', "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
    signal = mne.io.RawArray(data[tmp_signal], info)
    
    signal = signal.set_montage(ten_twenty_montage)

    signal.plot(scalings="auto")



def extract_powers(a:np.array, windowlength = 4*128, nc = 14, nt= 512, viz= False):
    data = a.reshape(a.shape[0], nc, nt)
    freqs, psd = signal.welch(data, 128, axis = 2)

    idx_delta = np.logical_and(freqs >= 1, freqs <= 4)
    idx_alpha = np.logical_and(freqs >= 8, freqs <= 13)
    idx_beta = np.logical_and(freqs >= 13, freqs <= 35)
    idx_gamma = np.logical_or(freqs >= 35, freqs >= 36 )
    idx_theta = np.logical_and(freqs >= 4, freqs <= 8)

    
    
    freq_res = freqs[2] - freqs[1] 
    
    all_powers = []
    all_powers_mean = []
    for i in range(len(psd)): # n 
        delta_signal = []
        alpha_signal = []
        beta_signal = []
        gamma_signal = []
        theta_signal = []

        for j in range((psd.shape[1])): #nc
            delta_power = simps(psd[i,j,idx_delta], dx=freq_res)
            alpha_power = simps(psd[i,j,idx_alpha], dx=freq_res)
            beta_power = simps(psd[i,j,idx_beta], dx=freq_res)
            gamma_power = simps(psd[i,j,idx_gamma], dx=freq_res)
            theta_power = simps(psd[i,j,idx_theta], dx=freq_res)

            total_power = simps(psd[i,j,:], dx=freq_res)

            delta_rel_power = delta_power/total_power
            alpha_rel_power = alpha_power/total_power
            beta_rel_power = beta_power/total_power
            gamma_rel_power = gamma_power/total_power
            theta_rel_power = theta_power/total_power

            delta_signal.append(delta_power)
            alpha_signal.append(alpha_power)
            beta_signal.append(beta_power)
            gamma_signal.append(gamma_power)
            theta_signal.append(theta_power)


        all_powers.append([*delta_signal, *alpha_signal,
                         *beta_signal, *gamma_signal,
                         *theta_signal])
        
        all_powers_mean.append([np.mean(delta_rel_power),np.mean(alpha_rel_power),
                         np.mean(beta_rel_power), np.mean(gamma_rel_power),
                         np.mean(theta_rel_power)])



    all_powers = np.array(all_powers)
    all_powers_mean = np.array(all_powers_mean)
    
    
    # Visualize the mean relative energy across all channels for each frequency band
    if (viz == True) and (len(a) == 1045):
        
        
        df = pd.DataFrame({"delta_avg_power": all_powers_mean[:,0],"alpha_avg_power": all_powers_mean[:,1],
                          "beta_avg_power": all_powers_mean[:,2],"gamma_avg_power": all_powers_mean[:,3],
                          "theta_avg_power": all_powers_mean[:,4],"label": labels.values})

        print(df)
        like_data = df[df["label"] == 1]
        dislike_data = df[df["label"] == 0]

        df["beta_avg_power"].shape

        f, axes = plt.subplots(5, 1, figsize=(20, 20), sharex=False)


        sns.lineplot(np.arange(0,1045),y=df["beta_avg_power"], hue = df["label"], ax=axes[0])
        sns.lineplot(np.arange(0,1045),y=df["delta_avg_power"], hue = df["label"], ax=axes[1])
        sns.lineplot(np.arange(0,1045),y=df["alpha_avg_power"], hue = df["label"], ax=axes[2])
        sns.lineplot(np.arange(0,1045),y=df["theta_avg_power"], hue = df["label"], ax=axes[3])
        sns.lineplot(np.arange(0,1045),y=df["gamma_avg_power"], hue = df["label"], ax=axes[4])

    return all_powers






def fourier_tf(data, fs = 128, viz =False, tmp_signal = None, tmp_channel= None):
    
    if isinstance(data, pd.DataFrame):
        tmp = np.absolute(rfftn(data.values.reshape(data.shape[0],14,512), axes=2 ))
    
    else:
        
        tmp = np.absolute(rfftn(data.reshape(data.shape[0],14,512), axes=2 ))
    
    fft_freq = rfftfreq(512, 1.0/fs)
    
    
    if viz == True:
        plt.plot(fft_freq, t[tmp_signal, tmp_channel,:])
        
    return tmp 






def extract_wavelet(data, level = 5):
    if isinstance(data, pd.DataFrame):
        data = data.values.reshape(data.shape[0],14,512)
        
    coeffs= pywt.wavedec(data.reshape(data.shape[0],14,512), "db4", axis=2, level = 5)


    cA5, cD5, cD4, cD3, cD2, cD1 = coeffs

    all_coeffs = np.concatenate([np.max(cA5, axis=2),np.min(cA5, axis=2) ,np.mean(cA5, axis=2), np.std(cA5, axis=2),
                                 np.max(cD5, axis=2),np.min(cD5, axis=2) ,np.mean(cA5, axis=2), np.std(cD5, axis=2),
                                 np.max(cD4, axis=2),np.min(cD4, axis=2) ,np.mean(cD4, axis=2), np.std(cD4, axis=2),
                                 np.max(cD3, axis=2),np.min(cD4, axis=2) ,np.mean(cD4, axis=2), np.std(cD3, axis=2),
                                 np.max(cD2, axis=2),np.min(cD2, axis=2) ,np.mean(cD2, axis=2), np.std(cD2, axis=2),
                                np.max(cD1, axis=2),np.min(cD1, axis=2) ,np.mean(cD1, axis=2), np.std(cD1, axis=2)], axis=1)
    
    
    all_coeffs_energy = np.concatenate([np.sum(np.abs(cA5**2), axis = 2), np.sum(np.abs(cD5**2), axis = 2),
                                       np.sum(np.abs(cD4**2), axis = 2),np.sum(np.abs(cD3**2), axis = 2),
                                       np.sum(np.abs(cD2**2), axis = 2),np.sum(np.abs(cD1**2), axis = 2)], axis=1)
    
    
    return all_coeffs_energy, all_coeffs
