"""Deterministic complete supervision passes with checkpointable boundaries."""
import numpy as np

class CoverageOrder:
    def __init__(self,nodes,seed,state=None):
        self.nodes=nodes;self.seed=seed;self.pass_index=0;self.cursor=0
        if state is not None:
            if state['nodes']!=nodes or state['seed']!=seed:raise ValueError('Sampler identity mismatch')
            self.pass_index=state['pass_index'];self.cursor=state['cursor']
        if not 0<=self.cursor<=nodes or self.pass_index<0:raise ValueError('Invalid cursor')
        self.order=np.random.default_rng(np.random.SeedSequence([seed,self.pass_index])).permutation(nodes)
    def take(self,batch):
        if self.cursor==self.nodes:
            self.pass_index+=1;self.cursor=0
            self.order=np.random.default_rng(np.random.SeedSequence([self.seed,self.pass_index])).permutation(self.nodes)
        end=min(self.nodes,self.cursor+batch)
        result=self.order[self.cursor:end];self.cursor=end
        return result
    def state(self):
        return dict(nodes=self.nodes,seed=self.seed,pass_index=self.pass_index,cursor=self.cursor)
