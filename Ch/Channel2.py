import numpy as np
import matplotlib.pyplot as plt
import scipy
from numpy.testing._private.nosetester import NoseTester
import math
import cmath
import scipy.interpolate
import scipy
from scipy import signal
from quantiphy import Quantity

SNRdb = 25
doppler_Frequency = 10
carrier_Freq = Quantity(100e6, 'Hz')
pl_Exponent = 4.8 #urban area : usually 2.7 to 3.5
ref_PL = np.power((4 * math.pi * 1 * carrier_Freq / scipy.constants.speed_of_light),2) # Path loss in FSPL with distance 1km
distance = 10 # in KM
delay = []
AoA = 100
AoD = 200

print(ref_PL)

#then variance_of_channel = -path_loss in db

# k = time
# n = subcarrier
# l = number of scatterers 
# m = number of paths
# Wavelength = Frequency / Speed of Light

def create_2D_Grid() :
    x = np.arange(50)
    y = np.arange(50)
    X,Y = np.meshgrid(x,y)
    
#def ddF(x) :
#    if x>0 
 
#def channel_Gain() :
    
#    channel_Gain = 
    
#    return 

# Calculate Path Loss using Reference Path Loss in db and distance in KM
def path_Loss(ref_PL,pl_Exponent, distance) :
    p_L = ref_PL + (10 * pl_Exponent * math.log(distance,10)) 
    return p_L



def get_Sigma() :
    variance_of_channel_in_db = -path_Loss(ref_PL,pl_Exponent,distance)
    sigma = np.sqrt(2/(4-math.pi) * np.sqrt(pow(10,0.1*variance_of_channel_in_db))) # scale parameter for rice distribution
    return sigma
    
def generate_Rayleigh_Channel() :
    amplitude = np.random.rayleigh(get_Sigma())
    phase = np.random.uniform(-math.pi, math.pi)
    channel_Gain = f"{amplitude * math.exp(1j*phase):.0f}"
    return channel_Gain
    
    
#def channel_Gain(attenuation,time_Delay) :

#    channel_Gain =  attenuation * cmath.exp(-1j * 2 * (cmath.pi) * carrier_Freq * time_Delay) # calculate channel gain
#    return channel_Gain

def calc_CIR(k,n,l) :
    
    CIR = generate_Rayleigh_Channel() * cmath.exp(2 * math.pi * doppler_Frequency * k) * signal.unit_impulse(100,delay) * signal.unit_impulse(100,AoA) * signal.unit_impulse(100,AoD)
    return CIR
   

def channel(signal,k,n,l) :
    convolved = np.convolve(signal,calc_CIR(k,n,l))
    signal_Power = np.mean(abs(convolved**2))
    noise_Power = signal_Power * 10 **(-SNRdb/10) #calculate noise Power based on signal power
    
    #Gaussian Noise
    noise = np.sqrt(noise_Power/2) * (np.random.randn(*convolved.shape)+1j*np.random.rndn(*convolved.shape))
    return convolved + noise