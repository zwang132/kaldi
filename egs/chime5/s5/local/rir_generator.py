import math
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool

def sinc(x):
    if x == 0:
        return 1.
    else:
        return math.sin(x)/x

def intround(x):
    if x >= 0:
        x = long(x+0.5)
    else:
        x = long(x-0.5)
    return x

class Rir_Generator():
    def __init__(self, c, fs, rr, ss, LL, beta_input, nSamples=None, microphone_type='ominidirectional', nOrder=-1, nDimension=3, angle=[0.,0.], isHighPassFilter=True):
        self.pi = 3.14159265358979323846
        self.c = c                                          # Sound velocity in m/s
        self.fs = fs                                        # Sampling frequency in Hz
        self.rr = rr                                        # Coordinates of the receivers in m
        self.nMicrophones = len(rr)                         # Number of microphones
        self.ss = ss                                        # Coordinates of the source in m
        self.LL = LL                                        # Room size in m
        self.V = self.LL[0] * self.LL[1] *self.LL[2]        # Volume of the room
        self.S = 2*(self.LL[0]*self.LL[2]+self.LL[1]*self.LL[2]+self.LL[0]*self.LL[1])   # Surface area of the room
        self.reverberation_time = 0                         # Reverberation time

        self.beta = np.zeros(6)                             # Reflection coefficients
        if len(beta_input) == 1:
            self.reverberation_time = beta_input[0]
            if self.reverberation_time != 0:
                alfa = 24*self.V*math.log(10.0)/(self.c*self.S*self.reverberation_time)
                assert (alfa <= 1), "The reflection coefficients cannot be calculated using the current room parameters."
                for i in range(6):
                    self.beta[i] = math.sqrt(1-alfa)
        else:
            for i in range(6):
                self.beta[i] = beta_input[i]

        if nSamples != None:                                # Number of samples
            self.nSamples = nSamples
        else:
            if len(beta_input) > 1:
                alpha = ((1-pow(self.beta[0],2))+(1-pow(self.beta[1],2)))*LL[1]*LL[2] +\
                        ((1-pow(self.beta[2],2))+(1-pow(self.beta[3],2)))*LL[0]*LL[2] +\
                        ((1-pow(self.beta[4],2))+(1-pow(self.beta[5],2)))*LL[0]*LL[1]
                self.reverberation_time = 24*math.log(10.0)*self.V/(self.c*alpha)
                if self.reverberation_time < 0.128:
                    self.reverberation_time = 0.128
            self.nSamples = int(self.reverberation_time*self.fs)

        self.microphone_type = microphone_type              # Type of microphones

        self.nOrder = nOrder                                # Maximum reflection order
        assert (self.nOrder >= -1), "Invalid input arguments."

        assert (nDimension == 2 or nDimension == 3), "Invalid input arguments."
        self.nDimension = nDimension                        # Room dimensions
        if self.nDimension == 2:
            self.beta[4] = 0.
            self.beta[5] = 0.

        if len(angle) == 1:                                 # Direction in which the microphones pointed
            self.angle = [angle[0],0.]                      # [azimuth, elevation] in radius
        elif len(angle) == 2:
            self.angle = angle
        else:
            self.angle = [0., 0.]

        self.isHighPassFilter = isHighPassFilter            # Whether to apply high-pass filter

    def sim_microphone(self, x, y, z, angle, mtype):
        typelist = ['b', 'h', 'c', 's']
        # Polar Pattern       rho
        # Bidirectional        0
        # Hypercardioid      0.25
        # Cardioid            0.5
        # Subcardioid        0.75
        # Ominidirectional     1
        if mtype in typelist:
            if mtype == 'b':
                rho = 0
            elif mtype == 'h':
                rho = 0.25
            elif mtype == 'c':
                rho = 0.5
            elif mtype == "s":
                rho = 0.75

            vartheta = math.acos(z/math.sqrt(pow(x,2)+pow(y,2)+pow(z,2)))             # Elevation between image and microphone
            varphi = math.atan2(y,x)                                                  # Azimuth between image and microphone

            gain = math.sin(self.pi/2-angle[1]) * math.sin(vartheta) * math.cos\
            (angle[0]-varphi) + math.cos(self.pi/2-angle[1]) * math.cos(vartheta)
            gain = rho + (1-rho) * gain                                               # Signal attenuation A(theta)
            return gain
        else:
            return 1

    def rir_generate(self, Fc, Tw):
        # Parameters of image method
        # Fc -- The cut-off frequency equals fs/2 - Fc is the normalized cut-off frequency
        # Tw -- The time width of the low-pass FIR equals 2 * Tw
        self.Tw = 2*intround(Tw*self.fs)
        self.Fc = Fc
        imp = np.zeros((self.nMicrophones, self.nSamples))

        # Parameters of high-pass filter
        W = 2*self.pi*100/self.fs        # The cut-off frequency equals 100 Hz
        self.R1 = math.exp(-W)
        self.B1 = 2*self.R1*math.cos(W)
        self.B2 = -self.R1 * self.R1
        self.A1 = -(1+self.R1)

        this_map = Pool().map
        imps = this_map(self.miccal, range(self.nMicrophones))
        for idx in range(self.nMicrophones):
            imp[idx] = imps[idx]
        if self.reverberation_time != 0:
            beta_hat = self.beta[0]
        else:
            beta_hat = 0.
        return imp, beta_hat

    def miccal(self, idMic):
        # Samples for receivers in x, y, z
        Tw = self.Tw
        cTs = float(self.c) / self.fs    # Resolution in direction
        s = np.array(self.ss) / cTs      # Samples for source
        L = np.array(self.LL) / cTs      # Samples for room size
        r = np.zeros(3)
        LPI = np.zeros(Tw)
        Rm = np.zeros(3)
        Rp_plus_Rm = np.zeros(3)
        refl = np.zeros(3)

        this_imp = np.zeros((self.nSamples))
        r[0] = self.rr[idMic][0] / cTs
        r[1] = self.rr[idMic][1] / cTs
        r[2] = self.rr[idMic][2] / cTs
        # Range of modes
        n1 = int(math.ceil(self.nSamples/(2*L[0])))
        n2 = int(math.ceil(self.nSamples/(2*L[1])))
        n3 = int(math.ceil(self.nSamples/(2*L[2])))
        for mx in range(-n1, n1+1):
            Rm[0] = 2*mx*L[0]
            for my in range(-n2, n2+1):
                Rm[1] = 2*my*L[1]
                for mz in range(-n3, n3+1):
                    Rm[2] = 2*mz*L[2]
                    for q in range(2):
                        # Distance between source image and the microphone in x
                        Rp_plus_Rm[0] = (1-2*q)*s[0] - r[0] + Rm[0]
                        refl[0] = pow(self.beta[0], abs(mx-q)) * pow(self.beta[1], abs(mx))
                        for j in range(2):
                            # Distance between source image and the microphone in y
                            Rp_plus_Rm[1] = (1-2*j)*s[1] - r[1] + Rm[1]
                            refl[1] = pow(self.beta[2], abs(my-j)) * pow(self.beta[3], abs(my))
                            for k in range(2):
                                # Distance between source image and the microphone in z
                                Rp_plus_Rm[2] = (1-2*k)*s[2] - r[2] + Rm[2]
                                refl[2] = pow(self.beta[4],abs(mz-k)) * pow(self.beta[5], abs(mz))
                                # Distance between source image and the microphone
                                dist = math.sqrt(pow(Rp_plus_Rm[0], 2) + pow(Rp_plus_Rm[1], 2) + pow(Rp_plus_Rm[2], 2))

                                if abs(2*mx-q)+abs(2*my-j)+abs(2*mz-k) <= self.nOrder or self.nOrder == -1:
                                    fdist = math.floor(dist)
                                    if fdist < self.nSamples:
                                        gain = self.sim_microphone(Rp_plus_Rm[0], Rp_plus_Rm[1], Rp_plus_Rm[2], self.angle, self.microphone_type[0]) \
                                               *refl[0]*refl[1]*refl[2]/(4*self.pi*dist*cTs)
                                        # Hanning-windowed low-pass filter
                                        for n in range(Tw):
                                            LPI[n] = 0.5 * (1 - math.cos(2*self.pi*((n+1-(dist-fdist))/Tw))) * self.Fc * sinc(self.pi*self.Fc*(n+1-(dist-fdist)-(Tw/2)))
                                        # Impulse response
                                        startPosition = int(fdist-(Tw/2)+1)
                                        for n in range(Tw):
                                            if startPosition+n >= 0 and startPosition+n < self.nSamples:
                                                this_imp[startPosition+n] += gain * LPI[n]

        if self.isHighPassFilter == True:
            # Applying high-pass filter
            Y = np.zeros(3)
            for idx in range(3):
                Y[idx] = 0
            for idx in range(self.nSamples):
                X0 = this_imp[idx]
                Y[2] = Y[1]
                Y[1] = Y[0]
                Y[0] = self.B1*Y[1] + self.B2*Y[2] + X0
                this_imp[idx] = Y[0] + self.A1*Y[1] + self.R1*Y[2]
        return this_imp

if __name__ == '__main__':
    r = Rir_Generator(340, 16000, [[0.2,1.57,0]], [0.5,0.5,0.5], [11.32,5.24,2.7], [0.4], int(0.4*16000))
    h, beta_hat = r.rir_generate(1,0.004)
