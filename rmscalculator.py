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
    print('usage: rmscalculator.py [file 1] [period start] [period end] [file 2] [period start] [period end]')
    print('example: rmscalculator.py test1.wav 1 5 test2.wav 11 15')

def dbCalculatorHelper(filename, t1, t2, t3, t4):
    if (t1 > t2 or t3 > t4):
        print('invalid input: t1 > t2 or t3 > t4')
        return
    [signal, sample_rate, channels] = audioBasicIO.readAudioFile(filename)
    print("shape = {}, dim = {}, sample_rate = {} channels = {}".format(signal.shape, signal.shape[1], sample_rate, channels))
    sampleNum = signal.shape[1]
    trackLen = sampleNum/sample_rate
    print('track length = %.2f seconds' % trackLen)

    if (t2 > trackLen or t4 > trackLen):
        print('invalid input: t2 > trackLen or t4 > trackLen')
        return
    if (channels > 4):
        print('audio file channels should be <= 4')
        return

    #fig, ax = plt.subplots()
    #fig.suptitle('Test Result', fontsize=14, fontweight='bold')
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    for i in range(channels):
        db1 = dbCalculator(signal[i][t1*sample_rate : t2*sample_rate], sample_rate)
        db2 = dbCalculator(signal[i][t3*sample_rate : t4*sample_rate], sample_rate)
        print('channel:',i+1)
        print("1st period = {0:.4f}, 2nd period = {1:.4f}, difference = {2:.4f}".format(db1, db2, abs(db1 - db2)))

        '''
        textstr = '\n'.join((
        r'1st period = %.4f dB' % (db1, ),
        r'2nd period = %.4f dB' % (db2, ),
        r'difference = %.4f dB' % (abs(db1 - db2), )))
        # place a text box in upper left in axes coords
        ax.text(0.05, 0.95 - 0.24*i, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
        '''
    #plt.show()

def dbCalculatorHelper2(filename1, filename2, t1, t2, t3, t4):
    if (t1 > t2 or t3 > t4):
        print('invalid input: t1 > t2 or t3 > t4')
        return
    [signal, sample_rate, channels] = audioBasicIO.readAudioFile(filename1)
    [signal2, sample_rate2, channels2] = audioBasicIO.readAudioFile(filename2)
    print("shape = {}, dim = {}, sample_rate = {} channels = {}".format(signal.shape, signal.shape[1], sample_rate, channels))
    sampleNum = signal.shape[1]
    trackLen = sampleNum/sample_rate
    print('track1 length = %.2f seconds' % trackLen)

    trackLen2 = signal2.shape[1]/sample_rate2
    print('track2 length = %.2f seconds' % trackLen2)

    if (t2 > trackLen or t4 > trackLen2):
        print('invalid input: t2 > trackLen or t4 > trackLen2')
        return
    if (channels > 4 or channels2 > 4):
        print('audio file channels should be <= 4')
        return
    if (channels != channels2):
        print('audio file channel number not match')
        return

    fig, ax = plt.subplots()
    fig.suptitle('Test Result', fontsize=14, fontweight='bold')
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    for i in range(channels):
        db1 = dbCalculator(signal[i][t1*sample_rate : t2*sample_rate], sample_rate)
        db2 = dbCalculator(signal2[i][t3*sample_rate2 : t4*sample_rate2], sample_rate2)

        textstr = '\n'.join((
        r'1st period = %.4f dB' % (db1, ),
        r'2nd period = %.4f dB' % (db2, ),
        r'difference = %.4f dB' % (abs(db1 - db2), )))
        # place a text box in upper left in axes coords
        ax.text(0.05, 0.95 - 0.24*i, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    plt.show()

def main():

    if(len(sys.argv) == 6):
        filename = sys.argv[1]
        t1 = int(sys.argv[2])
        t2 = int(sys.argv[3])
        t3 = int(sys.argv[4])
        t4 = int(sys.argv[5])
        print('filename = ' + filename)
        print("t1 = {}, t2 = {}, t3 = {} t4 = {}".format(t1, t2, t3, t4))
        dbCalculatorHelper(filename, t1, t2, t3, t4)
    elif(len(sys.argv) == 7):
        filename1 = sys.argv[1]
        t1 = int(sys.argv[2])
        t2 = int(sys.argv[3])
        filename2 = sys.argv[4]
        t3 = int(sys.argv[5])
        t4 = int(sys.argv[6])
        print('filename1 = ' + filename1)
        print("t1 = {}, t2 = {}".format(t1, t2))
        print('filename2 = ' + filename2)
        print("t3 = {} t4 = {}".format(t3, t4))
        dbCalculatorHelper2(filename1, filename2, t1, t2, t3, t4)
    else:
        showUsage()
        sys.exit()

if __name__== "__main__":
  main()
