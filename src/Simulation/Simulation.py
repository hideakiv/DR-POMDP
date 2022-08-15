import numpy as np

def runSimulation(prm,b0,epsilon,env,seed=None):
    ##  prm: contains policy
    if seed is not None:
        np.random.seed(seed)
    b = np.copy(b0)
    state = np.random.choice(prm.nS,p=b0)
    print(state)

    cumreward = 0
    discount = prm.beta
    while True:
        a = getAction(b,prm,epsilon)
        Pa, Pap, Ra = env.getPaRa(b,a) #Pa: true Pa, Pap: observed Pa #Ra: true Ra
        (nextstate,z) = sampleJZ(Pa,state,prm)
        cumreward += Ra[state,nextstate,z]*discount
        discount = discount * prm.beta
        state = nextstate
        b = updateb(b,Pap,z)
        print(b,'a',a,'z',z,'state',state,'cumreward',cumreward)
        if env.done(discount):
            break
    return cumreward

def runSimulationWithError(prm,b0,epsilon,env,seed=None):
    ##  prm: contains policy
    if seed is not None:
        np.random.seed(seed)
    b = np.copy(b0)
    bp= np.copy(b0)
    state = np.random.choice(prm.nS,p=b0)

    cumreward = 0
    discount = prm.beta
    while True:
        a = getAction(b,prm,epsilon)
        Pa, Pap, Ra = env.getPaRa(bp,a) #Pa: true Pa, Pap: observed Pa #Ra: true Ra
        (nextstate,z) = sampleJZ(Pa,state,prm)
        cumreward += Ra[state,nextstate,z]*discount
        discount = discount * prm.beta
        state = nextstate
        b = updateb(b,Pap,z)
        bp= updateb(bp,Pa,z)

        if env.done(discount):
            break
    return cumreward


def getAction(b,prm,epsilon):
    a = prm.getLBAction(b)
    return a

def sampleJZ(P,i,prm):
    elem = []
    prob = []
    for j in range(prm.nS):
        for z in range(prm.nZ):
            elem.append((j,z))
            if P[i,j,z] < 0:
                prob.append(0)
            else:
                prob.append(P[i,j,z])

    c = np.random.choice(len(elem),p=prob)
    return elem[c]

def updateb(b,Pa,z):
    newb = np.zeros(len(b))
    for i in range(len(b)):
        newb += Pa[i,:,z] * b[i]
    return newb / np.sum(newb)
