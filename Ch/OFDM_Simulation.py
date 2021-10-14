import numpy as np
import math
import cmath
import scipy
import matplotlib.pyplot as plt
from scipy import constants
from scipy.stats import rice
from quantiphy import Quantity
from scipy import signal



K = 64 # number of OFDM subcarriers
CP = K//4  # length of the cyclic prefix: 25% of the block
P = 8 # number of pilot carriers per OFDM block
pilotValue = 3+3j # The known value each pilot transmits

allCarriers = np.arange(K)  # indices of all subcarriers ([0, 1, ... K-1])

pilotCarriers = allCarriers[::K//P] # Pilots is every (K/P)th carrier.

# For convenience of channel estimation, let's make the last carriers also be a pilot
pilotCarriers = np.hstack([pilotCarriers, np.array([allCarriers[-1]])])
P = P+1

# data carriers are all remaining carriers
dataCarriers = np.delete(allCarriers, pilotCarriers)

print ("allCarriers:   %s" % allCarriers)
print ("pilotCarriers: %s" % pilotCarriers)
print ("dataCarriers:  %s" % dataCarriers)


doppler_Frequency = 10
carrier_Freq = Quantity(100e6, 'Hz')
carrier_Freq2 = 900
delay = 100
AoA = 180
AoD = 100
k = 1
CIR1 = 0+0j

def channel_Gain(attenuation,time_Delay) :
    
    channel_Gain = attenuation * cmath.exp(-1j * 2 * (cmath.pi) * carrier_Freq2 * time_Delay) # calculate channel gain
    return channel_Gain


impulse_Response = channel_Gain(1,0) + channel_Gain(1,1/(2*carrier_Freq2))

#print(f"{impulse_Response:.0f}")


value = f"{cmath.exp(1j * math.pi/2):.0f}"


#print(f"{cmath.exp(1j * math.pi/2):.0f}")
#value = cmath.exp(1j * math.pi/2)
#print(f"{value:.0f}")

#print(scipy.constants.speed_of_light)
#print(scipy.constants.c)

ref_PL = np.power((4 * math.pi * 1 * carrier_Freq / scipy.constants.speed_of_light),2)

distance = 10
pl_Exponent = 3.5

p_L = ref_PL + (10 * pl_Exponent * math.log(distance,10)) 

variance_of_channel_in_db = -p_L

#variance_of_channel = pow(10,0.1*variance_of_channel_in_db)

sigma = np.sqrt(2/(4-math.pi) * np.sqrt(pow(10,0.1*variance_of_channel_in_db))) # scale parameter for rice distribution

#print("path loss : ", p_L)
#print(variance_of_channel)
#print(sigma)

value1 = np.random.rayleigh(sigma)
value2 = np.random.uniform(-math.pi, math.pi)

value3 = f"{value1 * cmath.exp(1j*math.pi*value2):.3f}"
value4 = value1 * cmath.exp(1j*math.pi*value2)



#CIR = value1*cmath.exp(1j*2*math.pi*value2) * cmath.exp(2 * math.pi * doppler_Frequency * k) * signal.unit_impulse(200,delay) * signal.unit_impulse(200,AoA) * signal.unit_impulse(200,AoD)
#CIR2 = value1*cmath.exp(1j*2*math.pi*value2) * cmath.exp(2 * math.pi * doppler_Frequency * k) * signal.unit_impulse(200,delay)

#CIR = np.random.rayleigh(sigma) * cmath.exp(1j*math.pi*np.random.uniform(-math.pi,math.pi)) * cmath.exp(1j * 2 * math.pi * doppler_Frequency)

#print(CIR)

h_List = [0,0,0,0,0] # impulse list 
TOA = [0,10,20,30,40] # time of arrival, Delayed
AoA = [10,30,40,50,60]
AoD = [100,180,120,160,110]

CIR = [0,0,0,0,0]
CIR_List = [0,0,0,0,0] #CIR List
CIR_List2 = [0,0,0,0,0]

AoD_Impulse = [0,0,0,0,0]
AoA_Impulse = [0,0,0,0,0]
num_of_Path = len(TOA)

for i in range (0,num_of_Path) :  # Impulse Response for delayed time tau
    h_List[i]= signal.unit_impulse(50,TOA[i])
    
for i in range (0,num_of_Path) :  # Impulse Response for AoA
    AoA_Impulse[i]= signal.unit_impulse(185,AoA[i])  
    
for i in range (0,num_of_Path) :  # Impulse Response for AoD
    AoD_Impulse[i]= signal.unit_impulse(185,AoD[i])  

for i in range (0,num_of_Path) :
    CIR_List[i] = np.random.rayleigh(sigma) * cmath.exp(1j*math.pi*np.random.uniform(-math.pi,math.pi)) * cmath.exp(1j * 2 * math.pi * doppler_Frequency)
   
for i in range (0,num_of_Path) :
    CIR_List2[i] = CIR_List[i] * h_List[i][TOA[i]] *  AoD_Impulse[i][AoD[i]] * AoA_Impulse[i][AoA[i]] # Channel Gain * Delay * AoA * AoD

print(list)

for i in range (0,num_of_Path) :    
    CIR1 = CIR_List[i]+CIR1

print(CIR1)

mu = 4 # bits per symbol (i.e. 16QAM)
payloadBits_per_OFDM = len(dataCarriers)*mu  # number of payload bits per OFDM symbol

mapping_table = {
    (0,0,0,0) : -3-3j,
    (0,0,0,1) : -3-1j,
    (0,0,1,0) : -3+3j,
    (0,0,1,1) : -3+1j,
    (0,1,0,0) : -1-3j,
    (0,1,0,1) : -1-1j,
    (0,1,1,0) : -1+3j,
    (0,1,1,1) : -1+1j,
    (1,0,0,0) :  3-3j,
    (1,0,0,1) :  3-1j,
    (1,0,1,0) :  3+3j,
    (1,0,1,1) :  3+1j,
    (1,1,0,0) :  1-3j,
    (1,1,0,1) :  1-1j,
    (1,1,1,0) :  1+3j,
    (1,1,1,1) :  1+1j
}
for b3 in [0, 1]:
    for b2 in [0, 1]:
        for b1 in [0, 1]:
            for b0 in [0, 1]:
                B = (b3, b2, b1, b0)
                Q = mapping_table[B]
                plt.plot(Q.real, Q.imag, 'bo')
                plt.text(Q.real, Q.imag+0.2, "".join(str(x) for x in B), ha='center')

plt.show()

demapping_table = {v : k for k, v in mapping_table.items()}


channelResponse = np.array([0.1+0.2j, 0.4+0.8j, 0.3+0.3j])  # the impulse response of the wireless channel
H_exact = np.fft.fft(CIR_List2, K)
plt.plot(allCarriers, abs(H_exact))
plt.show()
SNRdb = 25  # signal to noise-ratio in dB at the receiver 


bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
print ("Bits count: ", len(bits))
print ("First 20 bits: ", bits[:20])
print ("Mean of bits (should be around 0.5): ", np.mean(bits))

def SP(bits):
    return bits.reshape((len(dataCarriers), mu))
bits_SP = SP(bits)
print ("First 5 bit groups")
print (bits_SP[:5,:])

def Mapping(bits):
    return np.array([mapping_table[tuple(b)] for b in bits])
QAM = Mapping(bits_SP)
print ("First 5 QAM symbols and bits:")
print (bits_SP[:5,:])
print (QAM[:5])

def OFDM_symbol(QAM_payload):
    symbol = np.zeros(K, dtype=complex) # the overall K subcarriers
    symbol[pilotCarriers] = pilotValue  # allocate the pilot subcarriers 
    symbol[dataCarriers] = QAM_payload  # allocate the pilot subcarriers
    return symbol
OFDM_data = OFDM_symbol(QAM)
print ("Number of OFDM carriers in frequency domain: ", len(OFDM_data))

def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)
OFDM_time = IDFT(OFDM_data)
print ("Number of OFDM samples in time-domain before CP: ", len(OFDM_time))

def addCP(OFDM_time):
    cp = OFDM_time[-CP:]               # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning
OFDM_withCP = addCP(OFDM_time)
print ("Number of OFDM samples in time domain with CP: ", len(OFDM_withCP))

def channel(signal):
    convolved = np.convolve(signal, CIR_List2)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb/10)  # calculate noise power based on signal power and SNR
    
    print ("RX Signal power: %.4f. Noise power: %.10f" % (signal_power, sigma2))
    
    # Generate complex noise with given variance
    noise = np.sqrt(sigma2/2) * (np.random.randn(*convolved.shape)+1j*np.random.randn(*convolved.shape))
    return convolved + noise
OFDM_TX = OFDM_withCP
OFDM_RX = channel(OFDM_TX)
plt.figure(figsize=(8,2))
plt.plot(abs(OFDM_TX), label='TX signal')
plt.plot(abs(OFDM_RX), label='RX signal')
plt.legend(fontsize=10)
plt.xlabel('Time'); plt.ylabel('$|x(t)|$');
plt.grid(True);
plt.show()

def removeCP(signal):
    return signal[CP:(CP+K)]
OFDM_RX_noCP = removeCP(OFDM_RX)

def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)
OFDM_demod = DFT(OFDM_RX_noCP)

def channelEstimate(OFDM_demod):
    pilots = OFDM_demod[pilotCarriers]  # extract the pilot values from the RX signal
    Hest_at_pilots = pilots / pilotValue # divide by the transmitted pilot values
    
    # Perform interpolation between the pilot carriers to get an estimate
    Hest_abs = scipy.interpolate.interp1d(pilotCarriers, abs(Hest_at_pilots), kind='linear')(allCarriers)
    Hest_phase = scipy.interpolate.interp1d(pilotCarriers, np.angle(Hest_at_pilots), kind='linear')(allCarriers)
    Hest = Hest_abs * np.exp(1j*Hest_phase)
    
    plt.plot(allCarriers, abs(H_exact), label='Correct Channel')
    plt.stem(pilotCarriers, abs(Hest_at_pilots), label='Pilot estimates')
    plt.plot(allCarriers, abs(Hest), label='Estimated channel via interpolation')
    plt.grid(True); plt.xlabel('Carrier index'); plt.ylabel('$|H(f)|$'); plt.legend(fontsize=10)
    plt.ylim(0,2)
    plt.show()
    
    return Hest
Hest = channelEstimate(OFDM_demod)

def equalize(OFDM_demod, Hest):
    return OFDM_demod / Hest
equalized_Hest = equalize(OFDM_demod, Hest)

def get_payload(equalized):
    return equalized[dataCarriers]
QAM_est = get_payload(equalized_Hest)
plt.plot(QAM_est.real, QAM_est.imag, 'bo');
plt.show()
def Demapping(QAM):
    # array of possible constellation points
    constellation = np.array([x for x in demapping_table.keys()])
    
    # calculate distance of each RX point to each possible point
    dists = abs(QAM.reshape((-1,1)) - constellation.reshape((1,-1)))
    
    # for each element in QAM, choose the index in constellation 
    # that belongs to the nearest constellation point
    const_index = dists.argmin(axis=1)
    
    # get back the real constellation point
    hardDecision = constellation[const_index]
    
    # transform the constellation point into the bit groups
    return np.vstack([demapping_table[C] for C in hardDecision]), hardDecision

PS_est, hardDecision = Demapping(QAM_est)
for qam, hard in zip(QAM_est, hardDecision):
    plt.plot([qam.real, hard.real], [qam.imag, hard.imag], 'b-o');
    plt.plot(hardDecision.real, hardDecision.imag, 'ro')

plt.show()

def PS(bits):
    return bits.reshape((-1,))
bits_est = PS(PS_est)

print ("Obtained Bit error rate: ", np.sum(abs(bits-bits_est))/len(bits))

