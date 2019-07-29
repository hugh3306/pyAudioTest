from __future__ import division
import sys
from numpy.fft import rfft, irfft
from numpy import argmax, sqrt, mean, absolute, arange, log10
import numpy as np
import matplotlib.pyplot as plt
import audioBasicIO


def rms_flat(a):
    """
    Return the root mean square of all the elements of *a*, flattened out.
    """
    return sqrt(mean(absolute(a)**2))


def find_range(f, x):
    """
    Find range between nearest local minima from peak at index x
    """
    for i in arange(x+1, len(f)):
        if f[i+1] >= f[i]:
            uppermin = i
            break
    for i in arange(x-1, 0, -1):
        if f[i] <= f[i-1]:
            lowermin = i + 1
            break
    return (lowermin, uppermin)

def freqCalculator(filename):
    [signal, sample_rate, channels] = audioBasicIO.readAudioFile(filename)
    print("shape = {}, dim = {}".format(signal.shape, signal.shape[1]))
    samp_rate = sample_rate
    chunk = signal.shape[1]
    data = signal[0][:]

    # mic sensitivity correction and bit conversion
    mic_sens_dBV = -47.0 # mic sensitivity in dBV + any gain
    mic_sens_corr = np.power(10.0,mic_sens_dBV/20.0) # calculate mic sensitivity conversion factor

    # (USB=5V, so 15 bits are used (the 16th for negatives)) and the manufacturer microphone sensitivity corrections
    data = ((data/np.power(2.0,15))*5.25)*(mic_sens_corr) 

    # compute FFT parameters
    f_vec = samp_rate*np.arange(chunk/2)/chunk # frequency vector based on window size and sample rate
    mic_low_freq = 100 # low frequency response of the mic (mine in this case is 100 Hz)
    low_freq_loc = np.argmin(np.abs(f_vec-mic_low_freq))
    fft_data = (np.abs(np.fft.fft(data))[0:int(np.floor(chunk/2))])/chunk
    fft_data[1:] = 2*fft_data[1:]

    max_loc = np.argmax(fft_data[low_freq_loc:])+low_freq_loc

    # plot
    plt.style.use('ggplot')
    plt.rcParams['font.size']=18
    fig = plt.figure(figsize=(13,8))
    ax = fig.add_subplot(111)
    plt.plot(f_vec,fft_data)
    ax.set_ylim([0,2*np.max(fft_data)])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [Pa]')
    ax.set_xscale('log')
    plt.grid(True)

    # max frequency resolution 
    plt.annotate(r'$\Delta f_{max}$: %2.1f Hz' % (samp_rate/(2*chunk)),xy=(0.7,0.92),\
                xycoords='figure fraction')

    # annotate peak frequency
    annot = ax.annotate('Freq: %2.1f'%(f_vec[max_loc]),xy=(f_vec[max_loc],fft_data[max_loc]),\
                        xycoords='data',xytext=(0,30),textcoords='offset points',\
                        arrowprops=dict(arrowstyle="->"),ha='center',va='bottom')
        
    #plt.savefig('fft_1kHz_signal.png',dpi=300,facecolor='#FCFCFC')
    #plt.show()
    a_weight_data_f = 20*np.log10(fft_data[1:]/0.00002)
    print(a_weight_data_f.mean())

def spl_flat(a):
    return 20*np.log10(np.sqrt(np.mean(np.absolute(a)**2)))

def dbCalculator(signal, sample_rate):
    data = signal/32767
    N = data.shape[0]
    T = 1/sample_rate
    np.set_printoptions(suppress=True)
    #t_vec = np.arange(N)*T # time vector for plotting
    #plt.plot(t_vec,data)
    #plt.show()
    return spl_flat(data)

def showUsage():
    print('usage: rmscalculator.py [file] [1st period start] [1st period end] [2nd period start] [2nd period end]')
    print('example: rmscalculator.py test.wav 1 5 11 15')

def dbCalculatorHelper(filename, t1, t2, t3, t4):
    if (t1 > t2 or t3 > t4):
        showUsage()
        return
    [signal, sample_rate, channels] = audioBasicIO.readAudioFile(filename)
    print("shape = {}, dim = {}, sample_rate = {} channels = {}".format(signal.shape, signal.shape[1], sample_rate, channels))
    sampleNum = signal.shape[1]
    trackLen = sampleNum/sample_rate
    print('track length = %.2f seconds' % trackLen)

    if (t2 > trackLen or t4 > trackLen):
        showUsage()
        return
    if (channels > 4):
        print('audio file channels should be <= 4')
        return

    fig, ax = plt.subplots()
    fig.suptitle('Test Result', fontsize=14, fontweight='bold')
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    for i in range(channels):
        db1 = dbCalculator(signal[i][t1*sample_rate : t2*sample_rate], sample_rate)
        db2 = dbCalculator(signal[i][t3*sample_rate : t4*sample_rate], sample_rate)

        textstr = '\n'.join((
        r'1st period = %.4f dB' % (db1, ),
        r'2nd period = %.4f dB' % (db2, ),
        r'difference = %.4f dB' % (abs(db1 - db2), )))
        # place a text box in upper left in axes coords
        ax.text(0.05, 0.95 - 0.24*i, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    plt.show()

def main():
    if (len(sys.argv) < 6):
        showUsage()
        sys.exit()
    filename = sys.argv[1]
    t1 = int(sys.argv[2])
    t2 = int(sys.argv[3])
    t3 = int(sys.argv[4])
    t4 = int(sys.argv[5])
    print('filename = ' + filename)
    print("t1 = {}, t2 = {}, t3 = {} t4 = {}".format(t1, t2, t3, t4))
    dbCalculatorHelper(filename, t1, t2, t3, t4)

if __name__== "__main__":
  main()
