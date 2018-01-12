import numpy as np
from hmmlearn import hmm

class HiddenMarkov(object):
    
    def __init__(self,states,observations):
        self.states=states
        self.observations=observations
        self.n_states = len(states)
        self.n_observations = len(observations)
        return


    def predict(self,start_probability,transition_probability,emission_probability):

        model=hmm.MultinomialHMM(n_components=self.n_states)
        model.startprob_=start_probability
        model.transmat_=transition_probability
        model.emissionprob_=emission_probability

        result=np.array([[1,0,1]]).T
        result1=np.array([[0,1,0]]).T
        box=model.predict(result1)

        print("The ball picked:", ", ".join([self.observations[i] for i in result1.T[0]]))
        print("The hidden box", ", ".join([self.states[i] for i in box]))
        print('The probablity is %f'%np.exp(model.score(result1)))

    def fit(self,observation_squence,lengths=None,max_step=10):
        model=hmm.MultinomialHMM(n_components=self.n_states,n_iter=max_step)
        model.fit(observation_squence,lengths=lengths)
        print(model.startprob_)
        print(model.transmat_)
        print(model.emissionprob_)
        print(np.exp(model.score(observation_squence)))

states = ["box 1", "box 2", "box3"]

observations = ["red", "white"]

start_probability = np.array([0.2, 0.4, 0.4])

transition_probability = np.array([
    [0.5, 0.2, 0.3],
    [0.3, 0.5, 0.2],
    [0.2, 0.3, 0.5]
])

emission_probability = np.array([
    [0.5, 0.5],
    [0.4, 0.6],
    [0.7, 0.3]
])

hm=HiddenMarkov(states,observations)
hm.predict(start_probability,transition_probability,emission_probability)
X2 = np.array([[0,1,0,1],[0,0,0,1],[1,0,1,1],[1,0,1,1]])
hm.fit(X2,[1,1,2],max_step=3000)
    # hm.fit([[0],[1],[0]])
