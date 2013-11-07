import math, threading, os, time

import nef

from ca.nengo.model import SimulationMode
from ca.nengo.model.impl import NetworkImpl
from ca.nengo.util import MU
from ca.nengo.math.impl import IndicatorPDF, ConstantFunction


from misc import HRLutils
from agent import memory, actionvalues

from nengo_decl import *


def qnetwork(stateN, stateD, state_encoders, actions, learningrate,
                 stateradius=1.0, Qradius=1.0, load_weights=None):
    net = nef.Network("QNetwork")
    with declarative_syntax(net):
        N = 50
        statelength = math.sqrt(2*stateradius**2)
        tauPSC = 0.007
        num_actions = len(actions)
        init_Qs = 0.0
        weight_save = 600.0 #period to save weights (realtime, not simulation time)

        #set up relays
        direct_mode('state_relay', 1, dimension=stateD)
        add_decoded_termination('state_relay', 'input', MU.I(stateD), .001, False)

        #create state population
        ensemble('state_pop', neurons=LIF(stateN),
                 dimensions=stateD,
                 radius=statelength,
                 encoders=state_encoders,
                )
        connect('state_relay', 'state_pop', filter=tauPSC)

        memory('saved_state', neurons=LIF(N * 4), dimension=stateD,
               inputscale=50,
               radius=stateradius,
               direct_storage=True)

        # N.B. the "." syntax refers to an ensemble created by the `memory` macro
        connect('state_relay', 'saved_state.target')

        ensemble('old_state_pop', neurons=LIF(stateN),
                 dimensions=stateD,
                 radius=statelength,
                 encoders=state_encoders)

        connect('saved_state', 'old_state_pop', filter=tauPSC)

        # mess with the intercepts ?
        for name in 'state_pop', 'old_state_pop':
            set_intercepts(name, IndicatorPDF(0, 1))

        fixMode('state_relay')
        fixMode('state_pop', ['default', 'rate'])
        fixMode('old_state_pop', ['default', 'rate'])

        #set up action nodes

        #XXX getting bogged down because I don't know Java API well enough :(
        


class QNetwork(NetworkImpl):
    def __init__(self, stateN, stateD, state_encoders, actions, learningrate,
                 stateradius=1.0, Qradius=1.0, load_weights=None):
        NetworkImpl.__init__(self)
        self.name = "QNetwork"
        net = nef.Network(self, seed=HRLutils.SEED, quick=False)
        
        N = 50
        statelength = math.sqrt(2*stateradius**2)
        tauPSC = 0.007
        num_actions = len(actions)
        init_Qs = 0.0
        weight_save = 600.0 #period to save weights (realtime, not simulation time)
        
        #set up relays
        state_relay = net.make("state_relay", 1, stateD, mode="direct")
        state_relay.fixMode()
        state_relay.addDecodedTermination("input", MU.I(stateD), 0.001, False)
        
        #create state population
        state_fac = HRLutils.node_fac()
        state_fac.setIntercept(IndicatorPDF(0,1))
            
        state_pop = net.make("state_pop", stateN, stateD, 
                              radius=statelength,
                              node_factory=state_fac,
                              encoders=state_encoders)
#                              eval_points=MU.I(stateD))
#        state_pop = net.make_array("state_pop", stateN/stateD, stateD,
#                                   node_factory=state_fac)
        state_pop.fixMode([SimulationMode.DEFAULT, SimulationMode.RATE])
        
        net.connect(state_relay, state_pop, pstc=tauPSC)
        
        #create population tied to previous state (to be used in learning)
        saved_state = memory.Memory("saved_state", N*4, stateD, inputscale=50, radius=stateradius,
                                    direct_storage=True)
        net.add(saved_state)
        
        net.connect(state_relay, saved_state.getTermination("target"))
        
        old_state_pop = net.make("old_state_pop", stateN, stateD, 
                              radius=statelength,
                              node_factory=state_fac,
                              encoders=state_encoders)
#                              eval_points=MU.I(stateD))
#        old_state_pop = net.make_array("old_state_pop", stateN/stateD, stateD,
#                                   node_factory=state_fac)
        old_state_pop.fixMode([SimulationMode.DEFAULT, SimulationMode.RATE])
        
        net.connect(saved_state, old_state_pop, pstc=tauPSC)
        
        #set up action nodes
        decoders = state_pop.addDecodedOrigin("init_decoders", [ConstantFunction(stateD,init_Qs)], "AXON").getDecoders()
        actionvals = actionvalues.ActionValues("actionvals", N, stateN, actions, learningrate, Qradius=Qradius, init_decoders=decoders)
        net.add(actionvals)
        
        decoders = old_state_pop.addDecodedOrigin("init_decoders", [ConstantFunction(stateD,init_Qs)], "AXON").getDecoders()
        old_actionvals = actionvalues.ActionValues("old_actionvals", N, stateN, actions, learningrate, Qradius=Qradius, init_decoders=decoders)
        net.add(old_actionvals)
        
        net.connect(state_pop.getOrigin("AXON"), actionvals.getTermination("state"))
        net.connect(old_state_pop.getOrigin("AXON"), old_actionvals.getTermination("state"))
        
        if load_weights != None:
            self.loadWeights(load_weights)
        
            #find error between old_actionvals and actionvals
        valdiff = net.make_array("valdiff", N, num_actions, node_factory = HRLutils.node_fac())
        net.connect(old_actionvals, valdiff, transform=MU.diag([2]*num_actions), pstc=tauPSC)
        net.connect(actionvals, valdiff, transform=MU.diag([-2]*num_actions), pstc=tauPSC)
            #doubling values to get a bigger error signal
        
            #calculate diff between curr_state and saved_state and use that to gate valdiff
        statediff = net.make_array("statediff", N, stateD, intercept=(0.2,1))
        net.connect(state_relay, statediff, pstc=tauPSC)
        net.connect(saved_state, statediff, transform=MU.diag([-1]*stateD), pstc=tauPSC)
        
        net.connect(statediff, valdiff, func=lambda x: [abs(v) for v in x], 
                    transform = [[-10]*stateD for _ in range(valdiff.getNeurons())], pstc=tauPSC)
        
        net.connect(valdiff, actionvals.getTermination("error"))
        
        #periodically save the weights
        class WeightSaveThread(threading.Thread):
            def __init__(self, func, prefix, period):
                threading.Thread.__init__(self)
                self.func = func
                self.prefix = prefix
                self.period = period
                
            def run(self):
                while True:
                    time.sleep(self.period)
                    self.func(self.prefix)
        wsn = WeightSaveThread(self.saveWeights, os.path.join("weights","tmp"), weight_save)
        wsn.start()
        
        self.exposeTermination(state_relay.getTermination("input"), "state")
        self.exposeTermination(old_actionvals.getTermination("error"), "error")
        self.exposeTermination(saved_state.getTermination("transfer"), "save_state")
        self.exposeOrigin(actionvals.getOrigin("X"), "vals")
        self.exposeOrigin(old_actionvals.getOrigin("X"), "old_vals")
        
    def saveWeights(self, prefix):
        self.getNode("actionvals").saveWeights(prefix)
        self.getNode("old_actionvals").saveWeights(prefix)
            
    def loadWeights(self, prefix):
        self.getNode("actionvals").loadWeights(prefix)
        self.getNode("old_actionvals").loadWeights(prefix)
