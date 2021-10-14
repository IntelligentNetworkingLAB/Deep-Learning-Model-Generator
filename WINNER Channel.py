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

#print ("allCarriers:   %s" % allCarriers)
#print ("pilotCarriers: %s" % pilotCarriers)
#print ("dataCarriers:  %s" % dataCarriers)


doppler_Frequency = Quantity(100e6, 'Hz')
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

#print("path loss :" , p_L)
#print("variance of channel in db : " , variance_of_channel_in_db)
#print("sigma: ", sigma)

value1 = np.random.rayleigh(sigma)
value2 = np.random.uniform(-math.pi, math.pi)

value3 = f"{value1 * cmath.exp(1j*math.pi*value2):.3f}"
value4 = value1 * cmath.exp(1j*math.pi*value2)

#print(value1)
#print(value2)



#CIR = value1*cmath.exp(1j*2*math.pi*value2) * cmath.exp(2 * math.pi * doppler_Frequency * k) * signal.unit_impulse(200,delay) * signal.unit_impulse(200,AoA) * signal.unit_impulse(200,AoD)
#CIR2 = value1*cmath.exp(1j*2*math.pi*value2) * cmath.exp(2 * math.pi * doppler_Frequency * k) * signal.unit_impulse(200,delay)

#CIR = np.random.rayleigh(sigma) * cmath.exp(1j*math.pi*np.random.uniform(-math.pi,math.pi)) * cmath.exp(1j * 2 * math.pi * doppler_Frequency)

#print(CIR)

h_List = [] # impulse list 
TOA = [0,10,20,30,40] # time of arrival, Delayed in micro sec


AoA = [10,30,40,50,60]
AoD = [100,180,120,160,110]

CIR = []
CIR_List = [] #CIR List
CIR_List2 = []

AoD_Impulse = []
AoA_Impulse = []
num_of_Path = len(TOA)

for i in range (0,num_of_Path) :  # Impulse Response for delayed time tau
    h_List.append(signal.unit_impulse(50,TOA[i]))
    
for i in range (0,num_of_Path) :  # Impulse Response for AoA
    AoA_Impulse.append(signal.unit_impulse(185,AoA[i]))  
    
for i in range (0,num_of_Path) :  # Impulse Response for AoD
    AoD_Impulse.append(signal.unit_impulse(185,AoD[i]))  

for i in range (0,num_of_Path) :
    CIR_List.append ((np.random.rayleigh(sigma) + np.random.uniform(-math.pi,math.pi)* 1j) * cmath.exp(1j*math.pi*np.random.uniform(-math.pi,math.pi)) * cmath.exp(1j * 2 * math.pi * doppler_Frequency))
   
for i in range (0,num_of_Path) :
    CIR_List2.append(CIR_List[i] * h_List[i][TOA[i]] *  AoD_Impulse[i][AoD[i]] * AoA_Impulse[i][AoA[i]]) # Channel Gain * Delay * AoA * AoD

print(CIR_List)
