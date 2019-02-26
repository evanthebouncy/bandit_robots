import scipy.io
import numpy as np
import random

mm = scipy.io.loadmat('1800_rrt.mat')['standard_selected_rewards']
M = []
for m in mm:
    if np.max(m) > -3.0:
        M.append(m)

M = np.array(M)

M_train = M[:400]
M_test  = M[400:]

print ("processed train / test split ")
print (len(M_train), len(M_test))

def gen_train_input(x):
    row_pad = np.ones(x.shape)
    partial_info = np.concatenate((np.array([x]), np.array([row_pad])), axis=0)

    n_ablate = np.random.randint(0, x.shape[0])
    n_ablate_idx = sorted(list(np.random.choice(list(range(x.shape[0])), n_ablate, replace=False)))
    partial_info[0][n_ablate_idx] = 0.0
    partial_info[1][n_ablate_idx] = 0.0
    return partial_info

def gen_train_data(n_batch):
    ret_in = []
    ret_out = []
    for _ in range(n_batch):
        r_row_idx = np.random.randint(0, M_train.shape[0])
        out = M_train[r_row_idx]
        inn = gen_train_input(out)

        ret_in.append(inn)
        ret_out.append(out)

    return np.array(ret_in), np.array(ret_out)

class Env:
    
    def __init__(self, test_row):
        self.truth = test_row
        self.seen = set([])

    def render(self):
        row_pad = np.ones(self.truth.shape)
        partial_info = np.concatenate((np.array([self.truth]), np.array([row_pad])), axis=0)

        ablate_idx = [x for x in range(self.truth.shape[0]) if x not in self.seen]
        partial_info[0][ablate_idx] = 0.0
        partial_info[1][ablate_idx] = 0.0
        return partial_info
        
    def reset(self):
        self.seen = set([])
        return self.render()

    def step(self, idx):
        """
            reward fraction of the max
        """
        self.seen.add(idx)
        return self.render(), (self.truth[idx] - min(self.truth)) / \
                              (max(self.truth) - min(self.truth))

def get_rollout(env, agent):

    cur_reward = 0.0
    cur_state = env.reset()
    ret = []

    while cur_reward < 1.0:
        action = agent.query(cur_state)
        nxt_state, reward = env.step(action)
        ret.append( (cur_state, action, reward, nxt_state) )
        cur_state = nxt_state
        cur_reward = reward

    return ret

class RandomAgent:

    def __init__(self):
        print ("hi im rand")
        self.seen = set([])

    def query(self, state):
        actions = list(set(range(162)) - self.seen)
        act = random.choice(actions)
        self.seen.add(act)
        return act

if __name__ == '__main__':
    partial_X, X = (gen_train_data(10))
    print (partial_X.shape)
    print (partial_X[0])

    print (X.shape)
    print (X[0])

    env = Env(M_test[0])
    ro = get_rollout(env, RandomAgent())
    print (len(ro))

