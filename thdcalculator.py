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

def rss_flat(a):
    """
    Return the root mean square of all the elements of *a*, flattened out.
    """
    return sqrt(np.sum(absolute(a)**2))

def getFreqLoc(f_vec, freq):
    loc = np.argmin(np.abs(f_vec-freq))
    return loc

def thdCalculator(signal, sample_rate, freq):

    chunk = signal.shape[0]
    # mic sensitivity correction and bit conversion
    mic_sens_dBV = -47.0 # mic sensitivity in dBV + any gain
    mic_sens_corr = np.power(10.0,mic_sens_dBV/20.0) # calculate mic sensitivity conversion factor

    # (USB=5V, so 15 bits are used (the 16th for negatives)) and the manufacturer microphone sensitivity corrections
    #data = ((data/np.power(2.0,15))*5.25)*(mic_sens_corr)
    data = (signal/np.power(2.0,15))

    # compute FFT parameters
    f_vec = sample_rate*np.arange(chunk/2)/chunk # frequency vector based on window size and sample rate
    mic_low_freq = 10 # low frequency response of the mic (mine in this case is 100 Hz)
    low_freq_loc = getFreqLoc(f_vec, mic_low_freq)
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
    plt.annotate(r'$\Delta f_{max}$: %2.1f Hz' % (sample_rate/(2*chunk)),xy=(0.7,0.92),\
                xycoords='figure fraction')

    # annotate peak frequency
    annot = ax.annotate('Freq: %2.1f'%(f_vec[max_loc]),xy=(f_vec[max_loc],fft_data[max_loc]),\
                        xycoords='data',xytext=(0,30),textcoords='offset points',\
                        arrowprops=dict(arrowstyle="->"),ha='center',va='bottom')

    #plt.show()
    
    #plt.savefig('fft_1kHz_signal.png',dpi=300,facecolor='#FCFCFC')
    print("low_freq = {}, max_freq = {}".format(f_vec[low_freq_loc], f_vec[max_loc]))
    #print("low_freq_loc = {}, max_loc = {}".format(low_freq_loc, max_loc))

    # calculate thd
    delta_loc = 3
    freq_rms = np.array([], dtype=np.float32)
    rms_fund = 0
    freq_step = freq
    target_freq = freq
    counter = 1
    while target_freq < sample_rate/2:
        low_loc = getFreqLoc(f_vec, target_freq - delta_loc)
        high_loc = getFreqLoc(f_vec, target_freq + delta_loc)
        #print("low_loc = {}, high_loc = {}".format(low_loc, high_loc))
        rms_loc = np.argmax(fft_data[low_loc:high_loc])+low_loc
        if (target_freq == freq):
            rms_fund = fft_data[rms_loc]
        else:
            freq_rms = np.append(freq_rms, fft_data[rms_loc])
        counter += 1
        target_freq = freq_step * counter
    #print(freq_rms[:30])
    thd = rss_flat(freq_rms)/rms_fund
    print("thd = {0:.3f} %".format(thd*100))

def thdHelper(filename, freq):
    [signal, sample_rate, channels] = audioBasicIO.readAudioFile(filename)
    print("shape = {}, dim = {}, sample_rate = {} channels = {}".format(signal.shape, signal.shape[1], sample_rate, channels))
    sampleNum = signal.shape[1]
    trackLen = sampleNum/sample_rate
    print('track length = %.2f seconds' % trackLen)
    data = signal[0][:]
    thdCalculator(data, sample_rate, freq)

def showUsage():
    print('usage: thdcalculator.py [file] [1st period start] [1st period end]')
    print('example: rmscalculator.py test.wav 1 5')

def main():
    if (len(sys.argv) < 5):
        showUsage()
        sys.exit()
    filename = sys.argv[1]
    freq = int(sys.argv[2])
    t1 = int(sys.argv[3])
    t2 = int(sys.argv[4])
    print('filename = ' + filename)
    print("freq = {}, t1 = {}, t2 = {}".format(freq, t1, t2))
    thdHelper(filename, freq)
    #dbCalculatorHelper(filename, t1, t2)

if __name__== "__main__":
  main()
