import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
# load data from csv from csv_data using pandas
df = pd.read_csv('csv_data\gravitational_wave_data.csv', delimiter=',', names=['time', 'strain'])
time_val = [float(x) for x in df['time'].values[1:]]
strain_val = [float(x) for x in df['strain'].values[1:]]
yf = fft(strain_val)
xf = fftfreq(len(strain_val))
plt.figure(figsize = (10, 10))
plt.stem(time_val, strain_val, 'r-')
plt.title('Gravitational Wave Strain Data')
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.grid()
plt.show()
y_abs = np.abs(yf)
plt.subplot(3, 1, 1);
#plot the fourier transform of the strain data
plt.plot(xf, y_abs, 'b')
plt.title('Gravitational Wave Strain Fourier Transform')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Strain')
plt.grid()
plt.show()
plt.subplot(3, 1, 1);
#plot the spectogram of the strain data
plt.specgram(strain_val, Fs=1/(time_val[1]-time_val[0]), NFFT=2048, noverlap=128)
plt.title('Gravitational Wave Strain Spectrogram')  
plt.show()