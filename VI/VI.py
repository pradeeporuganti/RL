import matplotlib.pyplot as plt
import numpy as np

class State():
    def __init__(self, amt_in_hand, value=0) -> None:
        self.amt_in_hand = amt_in_hand
        self.value = value

def sys_dyn(s, a, event):
    
    if s.id == 100:
        _r = 1
        _s_next = -1
    elif s.id != 100 and event == 'win':
        _s_next = min(s.id + a, 100)
        _r = 0
    elif s.id != 100 and event == 'lose':
        _s_next = max(0, s.id - a)
        _r = 0
    
    return _r, _s_next

def optimbellman(s, A, states, p_h):
    
    Q = []
    for a in A:
        _r_win, _s_next_win = sys_dyn(s, a, 'win')
        _r_lose, _s_next_lose = sys_dyn(s, a, 'lose')
        for _ in states:
            if _.id == _s_next_win:
                _v_win = _.value
            elif _.id == _s_next_lose:
                _v_lose = _.value
            else:
                pass
        Q.append(p_h*(_r_win + _v_win) + (1-p_h)*(_r_lose + _v_lose))

    return Q.index(max(Q)), max(Q)

def value_iter(states, p_h):
    _Delta = 1
    _theta = 0.00001

    pi_ = [None]*100
    while _Delta > _theta:
        _Delta = 0
        for s in states:
            v = s.value                                # store its current value
            _max_bet = max(s.id, 100-s.id)             # get the maximum bet availiable in this state
            A = list(np.arange(_max_bet+1))            # action space at the current state
            bet, s.value = optimbellman(s, A, states, p_h)
            pi_[s.id] = bet
            _Delta = max(v, s.value)
    return pi_

def main():
    p_h = 0.25      # probability of getting heads

    # state-sapce  # state 101 is terminal
    S = [State()]*101
    policy = value_iter(S, p_h)
    
    #plt.plot(policy)
    #plt.show()

if __name__ == '__main__':
    main()