#!/usr/bin/env python

import sys, os
import wave
import struct
from rir_generator import Rir_Generator as RG
import json
import numpy as np
from scipy import signal as sg
import scipy.io.wavfile as siw
import random
import math
from scipy.stats import truncnorm

def get_truncated_normal(low, upp):
    mean = (low+upp)/2.0
    sd = (upp-low)/5.0
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

class Rever_Speech():
    def __init__(self, audio_dir, json_file, rever_n, noise_dir, output_dir, micid):
        self.c = 340
        self.fs = 16000
        self.audio_dir = audio_dir
        self.output_dir = output_dir
        self.noise_dir = noise_dir
        self.micid = micid
        with open(json_file, 'r+') as f:
            conf = json.load(f)
            self.load_json(conf)
        self.Set_parameters(rever_n)

    def load_json(self, conf):
        assert (conf["receivers"].has_key(self.micid)), "Cannot find unit in configuration file"
        self.rr = [conf["receivers"][self.micid]]
        self.LL = conf["room_size"]
        self.beta_range = conf["beta"]
        self.seg = conf["seg_length"]
        self.ns = conf["noise_source"]
        self.SNR = conf["SNR"]

    def Set_parameters(self, rever_n):
        self.rever_n = bool(int(rever_n))
        self.piece_num = int(math.ceil(self.seg/15.0))
        self.V = self.LL[0]*self.LL[1]*self.LL[2]
        self.S = 2*(self.LL[0]*self.LL[1]+self.LL[1]*self.LL[2]+self.LL[2]*self.LL[0])
        self.beta_lb = 24*self.V*math.log(10.0)/(self.c*self.S)
        beta_gen = get_truncated_normal(self.beta_lb+np.finfo(float).eps, self.beta_range[1])
        self.beta = [beta_gen.rvs()]
        if self.rever_n == True:
            self.beta_nall = beta_gen.rvs(self.ns[1])

    def GRS(self, s, beta):
        r = RG(self.c, self.fs, self.rr, s, self.LL, beta)
        h, beta_hat = r.rir_generate(1,0.004)
        return h

    def read_wav(self, filename, upsample):
        fs, signal = siw.read(filename)
        t = signal.T * 1.0 / 2**15
        if upsample == True:
            le = sg.upfirdn([.1,.2,.3,.4,.5,.6,.7,.8,.9,1,.9,.8,.7,.6,.5,.4,.3,.2,.1],t[0],10)[9:-1]
            ri = sg.upfirdn([.1,.2,.3,.4,.5,.6,.7,.8,.9,1,.9,.8,.7,.6,.5,.4,.3,.2,.1],t[1],10)[9:-1]
        else:
            le = t[0]
            ri = t[1]
        return le, ri

    def Add_Noise(self, lre, rre, lnr, rnr):
        lnr = lnr[0:len(lre)]
        rnr = rnr[0:len(rre)]
        snr = random.randint(self.SNR[0], self.SNR[1])
        snr_r = 10**(snr/10.0)
        lre = np.asarray(lre)
        rre = np.asarray(rre)
        rl_c = np.mean(lre**2)/np.mean(lnr**2)
        rr_c = np.mean(rre**2)/np.mean(rnr**2)
        lnr = np.sqrt(rl_c/snr_r)*lnr
        rnr = np.sqrt(rr_c/snr_r)*rnr
        lre = lre + lnr
        rre = rre + rnr
        return lre, rre, snr

    def write_wav(self, filename, le, ri):
        data = np.vstack((le,ri)).T
        data = data * 2**15
        data = np.asarray(data, dtype = np.int16)
        siw.write(filename, self.fs, data)

    def Gen_chunk(self, le, ri):
        for i in range(int(math.ceil(len(le)/10.0/(self.seg*self.fs)))):
            print "Seg_number:", i+1
            lt = le[i*self.seg*self.fs*10:(i+1)*self.seg*self.fs*10]
            rt = ri[i*self.seg*self.fs*10:(i+1)*self.seg*self.fs*10]
            file_list = self.output_dir + '/' + os.path.basename(self.audio_dir).split('.')[0] + '_' + self.micid + '.txt'
            f = open(file_list, 'a')
            # Revebrant speech
            lre = sg.fftconvolve(lt, self.h[0], mode='same')
            rre = sg.fftconvolve(rt, self.h[0], mode='same')
            lre = sg.upfirdn([1],lre,1,10)
            rre = sg.upfirdn([1],rre,1,10)
            # Generate noise
            nn = random.randint(self.ns[0], self.ns[1])
            if nn != 0:
                lnr = np.zeros(self.seg * self.fs,)
                rnr = np.zeros(self.seg * self.fs,)
                for k in range(nn):
                    lt_n = []
                    rt_n = []
                    for p in range(self.piece_num):
                        # Load noise sources
                        num = random.randint(1, 10786)
                        noise_path = self.noise_dir + '/noise' + str(num) + '.wav'
                        le_n, ri_n = self.read_wav(noise_path, False)
                        lt_n.extend(le_n)
                        rt_n.extend(ri_n)
                    lt_n = lt_n[0:self.seg*self.fs]
                    rt_n = rt_n[0:self.seg*self.fs]
                    if self.rever_n == True:
                        lt_n = sg.fftconvolve(lt_n, self.hn_t[k][0], mode='same')
                        rt_n = sg.fftconvolve(rt_n, self.hn_t[k][0], mode='same')
                    lnr += np.asarray(lt_n)
                    rnr += np.asarray(rt_n)
                    lt_n = None
                    rt_n = None
                # Add noise
                lre, rre, snrr = self.Add_Noise(lre, rre, lnr, rnr)
                lnr = None
                rnr = None
                print "Noise_number:", nn, "SNR(dB):", snrr
            else:
                print "Noise_number:", nn
            lre = sum(abs(lt))/sum(abs(lre))*lre
            rre = sum(abs(rt))/sum(abs(rre))*rre
            name = os.path.basename(self.audio_dir).split('.')[0] + '_' + self.micid + '_' + str(i+1) + '.wav'
            f.write('file \'' + name + '\'\n')
            f.close()
            name = self.output_dir + '/' + os.path.basename(self.audio_dir).split('.')[0] + '_' + self.micid + '_' + str(i+1) + '.wav'
            # Normalization for int16 quantization
            if lre.max() >= (2**15-1)/2**15:
                lre = lre/ lre.max() * (2**15-1)/2**15
            if lre.min() <= -2**15:
                lre = lre/ abs(lre.min())
            if rre.max() >= (2**15-1)/2**15:
                rre = rre/ rre.max() * (2**15-1)/2**15
            if rre.min() <= -2**15:
                rre = rre/ abs(rre.min())
            # Write wav file
            self.write_wav(name, lre, rre)
            lre = None
            rre = None

    def Main_process(self):
        #Generate Rirs
        s = [random.uniform(0,self.LL[0]), random.uniform(0,self.LL[1]), 1.6]
        print "Source_location:", s, "Source_RT60:", self.beta[0]
        self.h = self.GRS(s, self.beta)
        if self.rever_n == True:
            self.hn_t = []
            nl_all = []
            for i in range(self.ns[1]):
                nl = [random.uniform(0,self.LL[0]), random.uniform(0,self.LL[1]), random.uniform(0,self.LL[2])]
                nl_all.append(nl)
                beta_n = [self.beta_nall[i]]
                hn = self.GRS(nl, beta_n)
                self.hn_t.append(hn)
            print "Max_noise_number(With_reverberation):",self.ns[1]
            print "Noise_location:", nl_all, "Noise_RT60:", self.beta_nall
        else:
            print "Max_noise_number(Without_reverberation):", self.ns[1]
        # Load wav file
        le, ri = self.read_wav(self.audio_dir, True)
        self.Gen_chunk(le, ri)


if __name__ == '__main__':
    rever_sp = Rever_Speech(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
    rever_sp.Main_process()
