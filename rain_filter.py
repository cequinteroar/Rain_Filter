import matplotlib.pyplot as plt
import numpy as np
import pyaudio
from scipy.io.wavfile import read
from scipy import fftpack
from scipy.signal import butter, lfilter, freqz

def butter_lowpass(cut, fs, order):
    return butter(order, cut, fs=fs)

def lowpass_filter(left, right, cut, fs, order):
    b, a = butter_lowpass(cut, fs, order)
    y_left = lfilter(b, a, left)
    y_right = lfilter(b, a, right)
    return y_left, y_right

def butter_bandpass(lowcut, highcut, fs, order):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def bandpass_filter(left, right, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    w, h = freqz(b, a, fs=fs, worN=2000)
    # graf_filter(w,h,order)
    y_left = lfilter(b, a, left)
    y_right = lfilter(b, a, right)
    return y_left, y_right

def rainfilter(audio, fs, order):
    # Processing variables
    newsnd = np.zeros(audio.shape);

    # Channel separation
    left = audio[:, 0]
    right = audio[:, 1]

    # Low Pass Filtering
    y_left, y_right = lowpass_filter(left, right, 4500.0, fs, order)

    # y_left, y_right = bandpass_filter(left, right, 1600.0, 11000.0, fs, order=order)
    
    newsnd[:,0] = y_left; 
    newsnd[:,1] = y_right; 
    return newsnd;

def graf_filter(w, h, order):
    plt.figure(1)
    plt.clf()
    plt.plot(w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')
    plt.show();
    return;

def freq_analysis(audio, s):
    # Processing variables
    CHUNK = 2048;
    ft = np.zeros(shape=(4, CHUNK))
    # Playing each channel
    left = audio[:, 0]
    #q.play_file(left, Fs, 1)

    right = audio[:, 1]
    #q.play_file(right, Fs, 1)

    # FFT block-wise
    a = 0
    for i in range(0, 4*CHUNK, CHUNK):
        ft[a] = fftpack.fft(left[i: i + CHUNK])
        a+=1

    # Plotting fft
    ft = ft/len(audio)
    ft = ft[:, 0:round(CHUNK/2)]
    magnitude = abs(ft)**2
    f_axis = np.linspace(0, round(s/2), round(CHUNK/2))
    print(s)
    print(f_axis)

    plt.title("Fourier Transform")
    plt.plot(f_axis, magnitude[0], "b", f_axis, magnitude[1], "ro-", f_axis, magnitude[2], "yo-", f_axis, magnitude[3], "cs-")
    plt.legend(('First chunk (0-1023)', 'Second chunk (1024-2047)', 'Third chunk (2048-3071)', 'Fourth chunk (3072-4095)'))
    plt.ylabel('Amplitude')
    plt.xlabel('Freq [Hz]')
    # plt.xlim(0, CHUNK)
    plt.show()
    return;

def load_file(path):
    return read(path);

def play_file(audio, sampling_rate, channels):
    p = pyaudio.PyAudio()

    # open audio stream

    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=sampling_rate,
                    output=True)
    # play. May repeat with different volume values (if done interactively)

    sound = (audio.astype(np.int16).tostring())
    stream.write(sound)

    # close stream and terminate audio object
    stream.stop_stream()
    stream.close()
    p.terminate()
    return;

def freq_response(audio, sampling_rate):
    left_channel = audio[:, 0];
    right_channel = audio[:, 1];
    length_in_s = audio.shape[0] / sampling_rate;
    time = np.arange(audio.shape[0]) / audio.shape[0] * length_in_s
    plt.subplot(2,1,1)
    plt.plot(time, left_channel, 'r')
    plt.xlabel("time, s [left channel]")
    plt.ylabel("signal, relative units")
    plt.subplot(2,1,2)
    plt.plot(time, right_channel, 'b')
    plt.xlabel("time, s [right channel]")
    plt.ylabel("signal, relative units")
    plt.tight_layout()
    plt.show()
    left_fft = np.fft.rfft(left_channel);
    right_fft = np.fft.rfft(right_channel);
    left_freq = np.fft.rfftfreq(left_channel.size, d=1./sampling_rate);
    right_freq = np.fft.rfftfreq(right_channel.size, d=1./sampling_rate);
    left_spectrum_abs = np.abs(left_fft);
    right_spectrum_abs = np.abs(right_fft);
    plt.title("Left and Right channel fft spectrum")
    plt.figure(1);
    plt.subplot(211);
    plt.plot(left_freq, left_spectrum_abs, 'r');
    plt.xlabel("frequency, Hz [left channel]");
    plt.ylabel("Amplitude, units");
    plt.subplot(212);
    plt.plot(right_freq, right_spectrum_abs, 'b');
    plt.xlabel("frequency, Hz [right channel]");
    plt.ylabel("Amplitude, units");
    plt.show();
    return;