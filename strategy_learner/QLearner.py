"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

#Name:Amogh Nalwaya, UserID:analwaya3
class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0

        self.rar = rar
        self.radr = radr
        self.alpha = alpha
        self.gamma = gamma

        self.q = np.zeros((num_states, num_actions))
        self.r = np.zeros((num_states, num_actions))

        self.t = np.zeros((num_states, num_actions, num_states))
        self.t_count = self.t.copy()

        self.dyna = dyna
        self.num_states = num_states
        self.num_actions = num_actions

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        max_q_index = self.q[s, :].argmax()

        if np.random.uniform() < self.rar:
            action = rand.randint(0, self.num_actions-1)
        else:
            action = max_q_index

        self.s = s
        self.a = action

        if self.verbose: print "s =", s,"a =",action
        return action

    def query(self, s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        #if self.verbose: print(self.q[s_prime, :])
        max_q_index = self.q[s_prime, :].argmax()
        #if self.verbose: print(max_q_index)
        if np.random.uniform() < self.rar:
            action = rand.randint(0, self.num_actions-1)
        else:
            action = max_q_index

        old_value = (1 - self.alpha) * self.q[self.s, self.a]
        disc_future_value = self.alpha * (r + (self.gamma * self.q[s_prime, max_q_index]))

        self.q[self.s, self.a] = old_value + disc_future_value

        self.rar = self.rar * self.radr

        if(self.dyna != 0):
            self.updateModel(self.s, self.a, s_prime, r)
            self.hallucinate()


        self.s = s_prime
        self.a = action
        if self.verbose: print "s =", s_prime,"a =",action,"r =",r

        return action

    def updateModel(self, s, a, s_prime, r):
        self.t_count[s, a, s_prime] = self.t_count[s, a, s_prime] + 1
        self.t = self.t_count / self.t_count.sum(axis=2, keepdims=True)
        self.r[s, a] = ((1 - self.alpha) * self.r[s, a]) + (self.alpha * r)

    def hallucinate(self):
        for iteration in range(self.dyna):
            s_rand = rand.randint(0, self.num_states - 1)
            a_rand = rand.randint(0, self.num_actions - 1)
            t_temp = self.t[s_rand, a_rand, :]
            s_prime = np.random.multinomial(100, t_temp).argmax()
            r_temp = self.r[s_rand, a_rand]

            max_q_index = self.q[s_prime, :].argmax()
            old_value = (1 - self.alpha) * self.q[s_rand, a_rand]
            disc_future_value = self.alpha * (r_temp + (self.gamma * self.q[s_prime, max_q_index]))

            self.q[s_rand, a_rand] = old_value + disc_future_value

    def author(self):
        return 'analwaya3'

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
