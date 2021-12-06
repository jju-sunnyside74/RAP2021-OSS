"""
Module to calcurate acoustic parameter using NumPy arrays

from https://
"""

import numpy as np
import scipy.stats as stats
from scipy import signal

# Import Users
import pyOssDebug as dbg


def T20(decayCurveNorm, fs):
    """Calculate T20

    :param decayCurveNorm: is the normalized decay curve
    :param fs: sample rate
    :return: T20
    """
    #T, nonLin = _reverberation(decayCurveNorm, -5, -25, fs)
    T, nonLin = _reverberation(decayCurveNorm, fs, -5, -25)
    T20 = T
    return T20, nonLin


def T30(decayCurveNorm, fs):
    """Calculate T30

    :param decayCurveNorm: is the normalized decay curve
    :param fs: sample rate
    :return: T30
    """

    #T, nonLin = _reverberation(decayCurveNorm, -5, -35, fs)
    T, nonLin = _reverberation(decayCurveNorm, fs, -5, -35)
    T30 = T
    return T30, nonLin


def RT60(decayCurveNorm, fs):
    """Calculate T60

    :param decayCurveNorm: is the normalized decay curve
    :param fs: sample rate
    :return: T60
    """

    #T, nonLin = _reverberation(decayCurveNorm, -5, -65, fs)
    T, nonLin = _reverberation(decayCurveNorm, fs, -5, -65)
    RT60 = T
    return RT60, nonLin


def EDT(decayCurveNorm, fs):
    """Calculate Early Decay Time (EDT)

    :param decayCurveNorm: is the normalized decay curve
    :param fs: sample rate
    :return: EDT
    """

    #T, nonLin = _reverberation(decayCurveNorm, 0, -10, fs)
    T, nonLin = _reverberation(decayCurveNorm, fs, 0, -10)
    EDT = T * 6
    return EDT, nonLin


def C50(IR, fs):
    """Calculate clarity for speech (C50)

    :param IR: impulse response
    :param fs: sample rate
    :return: C50
    """

    #C50 = _clarity(IR, 50, fs)
    C50 = _clarity(IR, fs, 50)
    return C50


def C80(IR, fs):
    """Calculate clarity for music (C80)

    :param IR: impulse response
    :param fs: sample rate
    :return: C80
    """

    #C80 = _clarity(IR, 80, fs)
    C80 = _clarity(IR, fs, 80)
    return C80


def D50(IR, fs):
    """Calculate definition (D50)

    :param IR: impulse response
    :param fs: sample rate
    :return: D50
    """

    #D50 = _definition(IR, 50, fs)
    D50 = _definition(IR, fs, 50)
    return D50


def centreTime(IR, fs):
    """Calculate the centre time from impulse response

    :param IR: impulse response
    :param fs: sample rate
    :return: centre time
    """
    if IR.ndim == 1:
        IR = IR[:, np.newaxis]

    t = np.repeat(np.linspace(0, np.size(IR, axis=0) / fs, np.size(IR, axis=0))[:, np.newaxis], np.size(IR, axis=1), axis=1)
    Ts = np.divide(np.sum(t * IR ** 2, axis=0), np.sum(IR ** 2, axis=0))
    if len(Ts) == 1:
        Ts = float(Ts)
    if Ts.ndim == 1:
        Ts = Ts[:, np.newaxis]
    return Ts


def lateralEnergyFraction(IR, IROmni, fs):
    """Calculate the lateral energy fraction from two impuleses

    :param IR: is an impulse from a figure of eight microphone
    :param IROmni: is an impulse from an omnidirection microphone
    :param fs: sample rate
    :return: lateral energy fraction
    """
    if IR.ndim == 1:
        IR = IR[:, np.newaxis]
    if IROmni.ndim == 1:
        IROmni = IROmni[:, np.newaxis]
    LF = np.divide(np.sum(IR[np.int64(5 / 1000 * fs):np.int64(80 / 1000 * fs)] ** 2, axis=0), np.sum(IROmni[np.int64(0 / 1000 * fs):np.int64(80 / 1000 * fs)] ** 2, axis=0))
    if LF.ndim == 1:
        LF = LF[:, np.newaxis]
    return LF


def strength(IR, IRSpeaker):
    """Calculate the strength from impulse responses

    :param IR: impulse response
    :param IRSpeaker: impulse response of the loudspeaker at 10 m in free field
    :return: strength
    """
    if IR.ndim == 1:
        IR = IR[:, np.newaxis]
    if IRSpeaker.ndim == 1:
        IRSpeaker = IRSpeaker[:, np.newaxis]

    G = 10 * np.log10(np.sum(IR ** 2, axis=0) / np.sum(IRSpeaker ** 2, axis=0))
    return G


#### IMPORTANT FUNCTIONS ####
'''
def decayCurve(sig, estimate, fs, noiseEnd=0):
    """Calculate the decay curve from a noise signal

    :param sig: noise signal
    :param estimate: the estimated reverb time (s)
    :param fs: sample rate
    :param noiseEnd: the time at which the noise stimuli stops
    :return:
    """

    decayCurvePa = exponential(sig, estimate / 40, fs)
    decayCurveSPL = 20 * np.log10(abs(decayCurvePa))
    decayCurveNorm = decayCurveSPL - np.max(decayCurveSPL[int(noiseEnd * fs):], axis=0)
    return decayCurveNorm
'''


def decayCurve(sig, estimate, fs, noiseEnd=0):
    """Calculate the decay curve from a noise signal

    :param sig: noise signal
    :param estimate: the estimated reverb time (s)
    :param fs: sample rate
    :param noiseEnd: the time at which the noise stimuli stops
    :return:
    """
    if estimate is None:
        estimate = sig.shape[0]/fs

    decayCurvePa = exponential(sig, estimate / 40, fs)
    decayCurveSPL = 20 * np.log10(abs(decayCurvePa))
    decayCurveNorm = decayCurveSPL - np.max(decayCurveSPL[int(noiseEnd * fs):], axis=0)
    return decayCurveNorm


def exponential(S, tau, fs):
    T = 1 / fs
    b = np.array([T / (T + tau)])
    b2 = np.array([1, -tau / (T + tau)])

    y = signal.lfilter(b, b2, S ** 2, axis=0)
    Ptau = np.sqrt(y)
    return Ptau


# EDT 계산

def _reverberation(decayCurveNorm, fs, reqDBStart=0, reqDBEnd=-10):
    """Calculate reverberation based on requirements for start and stop level

    :param decayCurveNorm: normalized decay curve
    :param fs: sample rate
    :param reqDBStart: start level for reverberation (is -5 for T60 and 0 for EDT)
    :param reqDBEnd: end level for reverberation (default: RT30 (-5 ~ -35dB)) (is -65 for RT60)
    :return: reveration
    """

    #func_name = "_reverbaration():"     # for debug

    if decayCurveNorm.ndim == 1:
        decayCurveNorm = decayCurveNorm[:, np.newaxis]

    # create return arrays based on number of decay curves
    T = np.zeros((np.size(decayCurveNorm, axis=1), 1))
    nonLinearity = np.zeros((np.size(decayCurveNorm, axis=1), 1))

    for i in range(np.size(decayCurveNorm, axis=1)):
        x_maxvalue = np.argmax(decayCurveNorm[:,i])
        #print(func_name,"x_maxvalue =", x_maxvalue, "y_value at x_max =", np.max(decayCurveNorm[:,i]))      # for debug
        #x_maxsearch = np.where(decayCurveNorm[:,i] == 0)[0][0]                                              # for debug
        #print(func_name,"x_maxsearch =", x_maxsearch, "y_value at x_max search =", decayCurveNorm[x_maxsearch,i])     # for debug

        try:
            #sample0dB = np.where(decayCurveNorm[:, i] < reqDBStart)[0][0]  # find first sample below 0 dB (Original Code no work)
            sample0dB = x_maxvalue + np.where(decayCurveNorm[x_maxvalue:, i] < reqDBStart)[0][0]  # find first sample below 0 dB
            #print(func_name, "reqDBStart=", reqDBStart, ", sample0dB=", sample0dB)  #for Debug
        except IndexError:
            #break
            #print("IndexError !!!  ", "reqDBStart=", reqDBStart, ", sample0dB=", sample0dB)  #for Debug
            raise ValueError("The is no level below {} dB".format(reqDBStart))

        try:
            sample10dB = sample0dB + np.where(decayCurveNorm[:, i][sample0dB:] <= reqDBEnd)[0][0]  # find first sample below -10dB
            #print(func_name, "reqDBEnd=", reqDBEnd, ",  sample10dB=", sample10dB)  #for Debug
        except IndexError:
            #break
            #print("IndexError !!!  ", "reqDBStart=", reqDBStart, ", sample0dB=", sample0dB)  #for Debug
            raise ValueError("The is no level below required {} dB".format(reqDBEnd))

        testDecay = decayCurveNorm[:, i][sample0dB:sample10dB]  # slice decaycurve to specific samples
        #dbg.dPlotAudio(fs, testDecay, "testDecay", "", "Time(sec)", "Amplitude") # for Debug

        slope, intercept, r_value, p_value, std_err = stats.linregress(np.linspace(sample0dB / fs, sample10dB / fs, np.size(testDecay, axis=0)), testDecay)  # calculate the slope and of signal nonlinearity
        nonLinearity[i] = np.round(1000 * (1 - r_value ** 2), 1)  # calculate the nonlinearity
        x = np.arange(-5, len(decayCurveNorm[:, i]) / fs + 5, 1 / 1000)  # create straight line with 1 ms resolution +- 5 secons
        y = intercept + slope * x  # generate line with slop of calculated slope

        T[i] = len(y[np.where(y <= reqDBStart)[0][0]:np.where(y <= reqDBEnd)[0][0]]) / 1000  # calculate reverberation with 1 ms resolution from generated linear line

    return T, nonLinearity

# 실제 RT60 계산에 사용 모듈
def T30_learning(decayCurveNorm, fs):
    """Calculate T30

    :param decayCurveNorm: is the normalized decay curve
    :param fs: sample rate
    :return: T30
    """
    # print("...T30 learning")

    #T, nonLin = _reverberation(decayCurveNorm, -5, -35, fs)
    s_0dB, s_10dB, s_20dB, s_30dB = _reverberation_P_learning(decayCurveNorm, fs, -5, -35)
    T, nonLin = _reverberation_T_learning(decayCurveNorm, fs, -5, -35)
    T30 = T
    return T30, nonLin, s_0dB, s_10dB, s_20dB, s_30dB


def _reverberation_P_learning(decayCurveNorm, fs, reqDBStart=-5, reqDBEnd=-35):
    """Calculate reverberation based on requirements for start and stop level

    :param decayCurveNorm: normalized decay curve
    :param fs: sample rate
    :param reqDBStart: start level for reverberation (is -5 for T60 and 0 for EDT)
    :param reqDBEnd: end level for reverberation (default: RT30 (-5 ~ -35dB)) (is -65 for RT60)
    :return:
       sample0dB
       sample10dB
       sample20dB
       sample30dB
    """

    #func_name = "_reverbaration():"     # for debug

    if decayCurveNorm.ndim == 1:
        decayCurveNorm = decayCurveNorm[:, np.newaxis]

    # create return arrays based on number of decay curves
    T = np.zeros((np.size(decayCurveNorm, axis=1), 1))
    nonLinearity = np.zeros((np.size(decayCurveNorm, axis=1), 1))

    for i in range(np.size(decayCurveNorm, axis=1)):
        x_maxvalue = np.argmax(decayCurveNorm[:,i])
        #print(func_name,"x_maxvalue =", x_maxvalue, "y_value at x_max =", np.max(decayCurveNorm[:,i]))      # for debug
        #x_maxsearch = np.where(decayCurveNorm[:,i] == 0)[0][0]                                              # for debug
        #print(func_name,"x_maxsearch =", x_maxsearch, "y_value at x_max search =", decayCurveNorm[x_maxsearch,i])     # for debug

        try:
            #sample0dB = np.where(decayCurveNorm[:, i] < reqDBStart)[0][0]  # find first sample below 0 dB (Original Code no work)
            sample0dB = x_maxvalue + np.where(decayCurveNorm[x_maxvalue:, i] < -5)[0][0]  # find first sample below 0 dB
            #print(func_name, "reqDBStart=", reqDBStart, ", sample0dB=", sample0dB)  #for Debug
        except IndexError:
            return 0, 0, 0, 0
            raise ValueError("The is no level-0 below {} dB".format(-5))

        try:
            sample10dB = sample0dB + np.where(decayCurveNorm[:, i][sample0dB:] <= -10)[0][0]  # find first sample below -10dB
            #print(func_name, "reqDBEnd=", reqDBEnd, ",  sample10dB=", sample10dB)  #for Debug
        except IndexError:
            return sample0dB, 0, 0, 0
            raise ValueError("The is no level-10 below required {} dB".format(-10))

        try:
            sample20dB = sample10dB + np.where(decayCurveNorm[:, i][sample10dB:] <= -20)[0][0]  # find first sample below -20dB
            #print(func_name, "reqDBEnd=", reqDBEnd, ",  sample10dB=", sample10dB)  #for Debug
        except IndexError:
            return sample0dB, sample10dB, 0, 0
            raise ValueError("The is no level-20 below required {} dB".format(-20))

        try:
            sample30dB = sample20dB + np.where(decayCurveNorm[:, i][sample20dB:] <= -30)[0][0]  # find first sample below -30dB
            #print(func_name, "reqDBEnd=", reqDBEnd, ",  sample10dB=", sample10dB)  #for Debug
        except IndexError:
            return sample0dB, sample10dB, sample20dB, 0
            raise ValueError("The is no level-30 below required {} dB".format(-30))

    #print("...sample0dB,  sample10dB = ", sample0dB, sample10dB)
    #print("...sample20dB, sample30dB = ", sample20dB, sample30dB)

    return sample0dB, sample10dB, sample20dB, sample30dB

def _reverberation_T_learning(decayCurveNorm, fs, reqDBStart=-5, reqDBEnd=-35):
    """Calculate reverberation based on requirements for start and stop level

    :param decayCurveNorm: normalized decay curve
    :param fs: sample rate
    :param reqDBStart: start level for reverberation (is -5 for T60 and 0 for EDT)
    :param reqDBEnd: end level for reverberation (default: RT30 (-5 ~ -35dB)) (is -65 for RT60)
    :return: reveration
    """

    #func_name = "_reverbaration():"     # for debug

    if decayCurveNorm.ndim == 1:
        decayCurveNorm = decayCurveNorm[:, np.newaxis]

    # create return arrays based on number of decay curves
    T = np.zeros((np.size(decayCurveNorm, axis=1), 1))
    nonLinearity = np.zeros((np.size(decayCurveNorm, axis=1), 1))

    for i in range(np.size(decayCurveNorm, axis=1)):
        x_maxvalue = np.argmax(decayCurveNorm[:,i])
        #print(func_name,"x_maxvalue =", x_maxvalue, "y_value at x_max =", np.max(decayCurveNorm[:,i]))      # for debug
        #x_maxsearch = np.where(decayCurveNorm[:,i] == 0)[0][0]                                              # for debug
        #print(func_name,"x_maxsearch =", x_maxsearch, "y_value at x_max search =", decayCurveNorm[x_maxsearch,i])     # for debug

        try:
            #sample0dB = np.where(decayCurveNorm[:, i] < reqDBStart)[0][0]  # find first sample below 0 dB (Original Code no work)
            sample0dB = x_maxvalue + np.where(decayCurveNorm[x_maxvalue:, i] < reqDBStart)[0][0]  # find first sample below 0 dB
            #print(func_name, "reqDBStart=", reqDBStart, ", sample0dB=", sample0dB)  #for Debug
        except IndexError:
            break
            raise ValueError("The is no level below {} dB".format(reqDBStart))

        try:
            sample10dB = sample0dB + np.where(decayCurveNorm[:, i][sample0dB:] <= reqDBEnd)[0][0]  # find first sample below -10dB
            #print(func_name, "reqDBEnd=", reqDBEnd, ",  sample10dB=", sample10dB)  #for Debug
        except IndexError:
            break
            raise ValueError("The is no level below required {} dB".format(reqDBEnd))

        testDecay = decayCurveNorm[:, i][sample0dB:sample10dB]  # slice decaycurve to specific samples
        #dbg.dPlotAudio(fs, testDecay, "testDecay", "", "Time(sec)", "Amplitude") # for Debug

        slope, intercept, r_value, p_value, std_err = stats.linregress(np.linspace(sample0dB / fs, sample10dB / fs, np.size(testDecay, axis=0)), testDecay)  # calculate the slope and of signal nonlinearity
        nonLinearity[i] = np.round(1000 * (1 - r_value ** 2), 1)  # calculate the nonlinearity
        x = np.arange(-5, len(decayCurveNorm[:, i]) / fs + 5, 1 / 1000)  # create straight line with 1 ms resolution +- 5 secons
        y = intercept + slope * x  # generate line with slop of calculated slope

        T[i] = len(y[np.where(y <= reqDBStart)[0][0]:np.where(y <= reqDBEnd)[0][0]]) / 1000  # calculate reverberation with 1 ms resolution from generated linear line

    return T, nonLinearity


def _clarity(IR, fs, t=50):
    """Calculate the clarity from impulse response

    :param IR: impulse response
    :param fs: sample rate
    :param t: is the defined shift from early to late reflections (is often 50 ms or 80 ms)(ms)
    :return: clarity
    """

    C = 10 * np.log10(np.sum(IR[0:np.int64((t / 1000) * fs)] ** 2, axis=0) / np.sum(IR[np.int64((t / 1000) * fs):] ** 2, axis=0))

    if C.ndim == 1:
        C = C[:, np.newaxis]
    return C


def _definition(IR, fs, t=50):
    """Calculate the defintion from impulse response

    :param IR: impulse response
    :param fs: sample rate
    :param t: is the defined shift from early to late reflections (is often 50 ms)(ms)
    :return: definition
    """

    D = np.sum(IR[0:np.int64(t / 1000 * fs)] ** 2, axis=0) / np.sum(IR ** 2, axis=0)

    if D.ndim == 1:
        D = D[:, np.newaxis]
    return D


def calcAcousticParam( data, decayCurveNorm, fs, use_rt60 = False, printout=False, label_text='' ):
	"""	Calculation Acoustic Parameter Values & Print out

	Parameters
	----------
    :param data: audio data array
    :param decayCurveNorm: Normalized decay curve data
    :param fs: sampling rate of the audio data
    :param RT60(option): Calculate real rt60 (True of False(=default))
    :param printout(option): Print All Acoustic Parameter
	:param fname: wave file path & name or struct_format_chunk

	Returns
	--------
	:param CAcousticParam: structure of Acoustic Parameters
	"""
	# Calculation Acoustic Parameters
	data_EDT, nonLin_EDT = EDT(decayCurveNorm, fs)
	data_t20, nonLin_t20 = T20(decayCurveNorm, fs)
	data_t30, nonLin_t30 = T30(decayCurveNorm, fs)
	if use_rt60 is True:
		data_t60, nonLin_t60 = RT60(decayCurveNorm, fs)
	else:
		data_t60 = data_t30
	data_C50 = C50(data, fs)
	data_C80 = C80(data, fs)
	data_D50 = D50(data, fs)

	if printout is True:
		print( "Label: ", label_text)
		print( " - Decay Time  0 ~ -10dB = ", data_EDT[0][0]/6)	# for Debug
		print( " - Decay Time -5 ~ -25dB = ", data_t20[0][0]/3)	# for Debug
		print( " - Decay Time -5 ~ -35dB = ", data_t30[0][0]/2)	# for Debug
		print( " - EDT = ", data_EDT[0][0])						# for Debug
		print( " - T20 = ", data_t20[0][0])         			# for Debug
		print( " - T30 = ", data_t30[0][0])         			# for Debug
		if RT60 is True:
			print( " - RT60(Real) = ", data_t60[0][0])			# for Debug
		else:
			print( " - RT60(=T30) = ", data_t60[0][0]) 			# for Debug
		print( " - C50 = ", data_C50)         			# for Debug
		print( " - C80 = ", data_C80)         			# for Debug
		print( " - D50 = ", data_D50)         			# for Debug

	CAcousticParam = CAcousticParameter(data_t60, data_EDT, data_D50, data_C50, data_C80)
	return CAcousticParam

class CAcousticParameter:
    def __init__(self, RT60, EDT, D50, C50, C80):
        self.RT60 = RT60[0][0]

        self.EDT = EDT[0][0]
        self.D50 = D50
        self.C50 = C50
        self.C80 = C80
