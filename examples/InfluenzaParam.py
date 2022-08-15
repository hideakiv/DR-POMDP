import numpy as np
import math
import scipy.stats

class Influenza:
    def __init__(self,nT,nA,nZ):
        self.discount = 0.95
        self.values = -1.0
        self.states = []
        for i in range(nT):
            self.states.append(str(i)+"-0")
            self.states.append(str(i)+"-1")
        self.actions = []
        for i in range(nA):
            self.actions.append(str(i))
        self.observations = []
        for i in range(nZ):
            self.observations.append(str(i))

        N = len(self.states)
        M = len(self.observations)
        L = len(self.actions)

        self.T = np.zeros((L,N,N))
        self.O = np.zeros((L,N,M))
        self.R = np.zeros((L,N,N,M))

        self.start = []
        self.start = np.zeros(N)
        self.start[0] = 0.5
        self.start[1] = 0.5

        #larger the a, stronger policy
        for a in range(L):
            for t in range(nT):
                x = 3.0 - 0.5*a + math.cos(t/nT)
                y = 2.0 + 0.5*a - math.cos(t/nT)
                p11 = 1/(1+math.exp(-x))
                p12 = 1-p11
                p22 = 1/(1+math.exp(-y))
                p21 = 1-p22
                self.T[a,2*t-2,2*t] = p11
                self.T[a,2*t-2,2*t+1] = p12
                self.T[a,2*t-1,2*t] = p21
                self.T[a,2*t-1,2*t+1] = p22
        #separate the probability with
        #0: less than 0
        #1~M-2: divide between 0~10
        #M-1: greater than 10
        rightPoints = np.zeros(M-1)
        for m in range(M-1):
            rightPoints[m] = m*10/(M-2)
        for a in range(L):
            for t in range(nT):
                mean1 = 1.597 + 3.349*math.cos(2*math.pi*t/nT) + 0.517*math.sin(2*math.pi*t/nT)
                std1  = math.sqrt(29.295 - mean1**2)
                mean2 = 0.551 + 0.286*math.cos(2*math.pi*t/nT) + 0.053*math.sin(2*math.pi*t/nT)
                std2  = math.sqrt(max(0.146 - mean2**2,0.001))

                self.O[a,2*t,0] = scipy.stats.norm.cdf(0,mean1,std1)
                self.O[a,2*t+1,0] = scipy.stats.norm.cdf(0,mean2,std2)
                for z in range(1,M-1):
                    self.O[a,2*t,z] = scipy.stats.norm.cdf(rightPoints[z],mean1,std1)-scipy.stats.norm.cdf(rightPoints[z-1],mean1,std1)
                    self.O[a,2*t+1,z] = scipy.stats.norm.cdf(rightPoints[z],mean2,std2)-scipy.stats.norm.cdf(rightPoints[z-1],mean2,std2)
                self.O[a,2*t,M-1] = scipy.stats.norm.sf(10,mean1,std1)
                self.O[a,2*t+1,M-1] = scipy.stats.norm.sf(10,mean2,std2)

        #R is cost because values = -1
        for a in range(L):
            for t in range(nT):
                for z in range(M-1):
                    self.R[a,2*t-2,2*t,z]   = -rightPoints[z] - 1*a/L
                    self.R[a,2*t-2,2*t+1,z] = -rightPoints[z] - 1*a/L
                    self.R[a,2*t-1,2*t,z]   = -rightPoints[z] - 1*a/L
                    self.R[a,2*t-1,2*t+1,z] = -rightPoints[z] - 1*a/L
                self.R[a,2*t-2,2*t,M-1]   = -15 - 1*a/L
                self.R[a,2*t-2,2*t+1,M-1] = -15 - 1*a/L
                self.R[a,2*t-1,2*t,M-1]   = -15 - 1*a/L
                self.R[a,2*t-1,2*t+1,M-1] = -15 - 1*a/L

