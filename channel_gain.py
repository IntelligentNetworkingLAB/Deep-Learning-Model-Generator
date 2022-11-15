import numpy as np
import math
import cmath
from quantiphy import Quantity

carrier_Freq = Quantity(100e6, 'Hz')

carrier_Freq2 = 900

def channel_Gain(attenuation,time_Delay) :
    
    channel_Gain = attenuation * cmath.exp(-1j * 2 * math.pi * carrier_Freq * time_Delay) # calculate channel gain
    return channel_Gain


impulse_Response = channel_Gain(1,0) + channel_Gain(1,1/2*carrier_Freq)


print(2*carrier_Freq2 * 1/2*carrier_Freq2)
