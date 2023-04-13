import pandas as pd
import numpy as np
import scipy
from scipy.io.wavfile import write

FSH = 8_820_000     # sampling frequency high frequency(Hz)
FSL = 44_100        # sampling frequency low frequency(Hz)
PERIOD = 3

def message(f,t):
    if hasattr(f,"__len__"):
        message = np.sum([np.sin(2*np.pi*fi*t) for fi in f],axis=0)
    else:
        message = np.sin(2*np.pi*f*t)
    return message

def QAM_modulation(message1,Fc,t,message2 = [],a1=0.5,a2=0.5,carrier_dev=0, transmission_dev=0):
    if hasattr(message2,"__len__") and len(message2)>0:
        if len(message1)>len(message2):
            message2.extend(np.zeros(len(message1)-len(message2)))
        elif len(message2)>len(message1):
            message1.extend(np.zeros(len(message2)-len(message1)))
    else:
        message2 = np.zeros_like(message1)

    Icarrier = np.sin(2*np.pi*Fc*t + transmission_dev*np.pi/2) # Carrier in phase
    Qcarrier = np.sin(2*np.pi*Fc*t + (1 + carrier_dev + transmission_dev)*np.pi/2) # Carrier in quadrature

    qam = (1 + a1*message1) * Icarrier + (1 + a2*message2) * Qcarrier
    return qam

def QAM_demodulation(Fc,t,a1,a2,signal,fcutoff=2_000,order=4):
    # Demodulate the QAM signal using a local oscillator
    local_oscilator = np.sin(2*np.pi*Fc*t) # Carrier in phase
    local_oscilator_q = np.sin(2*np.pi*Fc*t + np.pi/2) # Carrier in quadrature

    demod1 = a1 * signal * local_oscilator  # demodulate message1 signal
    demod2 = a2 * signal * local_oscilator_q  # demodulate message2 signal
    

    # Apply low-pass filters to remove high-frequency components
    wcutoff = 2*np.pi*fcutoff  # cutoff frequency for low-pass filter (radians/s)
    b, a = scipy.signal.butter(order, wcutoff, fs=FSH, btype='low', analog=False)  # filter coefficients

    demod1 = scipy.signal.lfilter(b, a, demod1)  # apply low-pass filter to demodulated message1 signal
    demod2 = scipy.signal.lfilter(b, a, demod2)  # apply low-pass filter to demodulated message2 signal

    return demod1.astype('float32'),demod2.astype('float32')


m2_f = "300,800"
m1_f = "100,500"
carrier_f = 800_000
m1_factor = 0.5
m2_factor = 0.5
carrier_dev = 0
phase_dev = 0.2
snr = 10
filter_freq = 2000
filter_order = 5

t = np.arange(0, PERIOD, 1/FSH)

# ------------------------------- Creating messages -------------------------------
df_message1 = pd.DataFrame(columns=["message","time"])
if type(m1_f) is str:
    f = np.array(m1_f.split(',')).astype(float)
else:
    f = m1_f
df_message1["message"] = message(f,t)
df_message1["time"] = t

df_message2 = pd.DataFrame(columns=["message","time"])
if type(m2_f) is str:
    f = np.array(m2_f.split(',')).astype(float)
else:
    f = m2_f
df_message2["message"] = message(f,t)
df_message2["time"] = t

# ------------------------------- Sound saving -------------------------------
amplitude = np.iinfo(np.int16).max
m1 = df_message1['message'][::int(FSH/FSL)]
data1 = m1 / np.max(np.abs(m1)) * amplitude
write("message1.wav", FSL, data1.astype(np.int16))

m2 = df_message2['message'][::int(FSH/FSL)]
data2 = m2 / np.max(np.abs(m2)) * amplitude
write("message2.wav", FSL, data2.astype(np.int16))

#------------------------------- QAM Modulation -------------------------------
df_qam = pd.DataFrame(columns=["message","time"])
df_qam["time"] = df_message1["time"] if len(df_message1["time"]) > len(df_message2["time"]) else df_message2["time"]
df_qam["message"] = QAM_modulation(df_message1["message"],carrier_f,df_qam["time"],message2=df_message2["message"],
                                    a1=m1_factor,a2=m2_factor,
                                    carrier_dev=carrier_dev, transmission_dev=phase_dev)

noise_std = np.sqrt(np.sum(df_qam["message"]**2) / (len(df_qam["message"]) * 10**(snr/10)))  # calculate noise standard deviation
noise = noise_std * (2*np.random.rand(len(df_qam["message"])) - 1)  # generate white noise
df_qam["message_noisy"] = df_qam["message"] + noise  # add noise to QAM signal

# ------------------------------- Demodulation -------------------------------
demodulated_m1 = df_qam.copy(deep=True).drop(columns=['message_noisy'])
demodulated_m2 = df_qam.copy(deep=True).drop(columns=['message_noisy'])
demodulated_m1['message'],demodulated_m2['message'] = QAM_demodulation(carrier_f,df_qam['time'],
                            m1_factor,m2_factor,df_qam['message_noisy'],
                            fcutoff=filter_freq,order=filter_order)

# ------------------------------- Sound saving -------------------------------
amplitude = np.iinfo(np.int16).max
m1 = demodulated_m1['message'][::int(FSH/FSL)]
data1 = m1 / np.max(np.abs(m1)) * amplitude
write("demod1.wav", FSL, data1.astype(np.int16))

m2 = demodulated_m2['message'][::int(FSH/FSL)]
data2 = m2 / np.max(np.abs(m2)) * amplitude
write("demod2.wav", FSL, data2.astype(np.int16))
