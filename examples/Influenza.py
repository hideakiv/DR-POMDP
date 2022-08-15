import numpy as np
import pickle
import time
import InfluenzaParam
from Parameter_setup import Parser, Params, Params_DR, Params_Standard
from HSVI import HSVI
from Simulation.Environment import *
from Simulation import Simulation

def main(parser,folder):

    pprm = Params_Standard.Params_Standard('POMDP',parser)

    prm = Params.Params('foo',parser)
    cap = np.zeros((prm.nA,prm.nS,prm.nS*prm.nZ))
    for a in range(prm.nA):
        for i in range(prm.nS):

            if a==0 and i==0:
                for j in range(prm.nS):
                    for z in range(prm.nZ):
                        jz = j*prm.nZ+z
                        cap[a,i,jz] = 0.05


    dprm = Params_DR.Params_DR('DRPOMDP',parser,cap=cap)

    b0 = parser.start
    epsilon = 1.0

    time_start = time.clock()
    #
    HSVI.HSVI(b0,epsilon,pprm,folder)
    #
    time_elapsed = (time.clock() - time_start)
    print('POMDP time elapsed:',time_elapsed)
    time_start = time.clock()
    #
    HSVI.HSVI(b0,epsilon,dprm,folder)
    #
    time_elapsed = (time.clock() - time_start)
    print('DRPOMDP time elapsed:',time_elapsed)

    return pprm, dprm

if __name__ == '__main__':
    fdir = './Experiments/InfluenzaPolicy/'
    folder = fdir

    parser = InfluenzaParam.Influenza(5,4,4)

    pprm, dprm = main(parser,folder)

    with open(folder+'pprm.pkl','wb') as output:
        pprm.save(folder)
        pickle.dump(pprm, output, pickle.HIGHEST_PROTOCOL)

    with open(folder+'dprm.pkl','wb') as output:
        dprm.save(folder)
        pickle.dump(dprm, output, pickle.HIGHEST_PROTOCOL)


    """
    with open(folder+'pprm.pkl', 'rb') as input:
        pprm = pickle.load(input)
        pprm.load(folder)

    with open(folder+'dprm.pkl', 'rb') as input:
        dprm = pickle.load(input)
        dprm.load(folder)
    """


    env = RegularEnv(dprm,1.0) # 

    preward = Simulation.runSimulation(pprm,parser.start,1.0,env)
    dreward = Simulation.runSimulation(dprm,parser.start,1.0,env)

    print(preward,dreward)
