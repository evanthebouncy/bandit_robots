from bay_opt import bay_sample
import numpy as np
from peak_map import gen_XrXp, rank_inputs, gen_params, make_Xr
import random
from model import Compl
from tqdm import tqdm

# an ON Policy training procedure
def get_policy_ob_rollout(peaks, compl, n_draws, epi):
    xx = np.random.random((1,))
    yy = np.array([peaks(x) for x in xx])
    
    for i in range(n_draws - 1):
        # draw at random
        if np.random.random() < epi:
            xx_new = np.random.random()
        # get the most informed
        else:
            mu, sig = bay_sample(compl, xx, yy)
            xx_new = np.argmax(sig) / len(sig)

        yy_new = peaks(xx_new)
        xx = np.array(list(xx) + [xx_new])
        yy = np.array(list(yy) + [yy_new])

    output_x = np.random.random()
    output_y = peaks(output_x)

    return xx, yy, output_x, output_y

def gen_batch_policy_ob_data(compl_model, n_pts, n_batch, epi):
    input_xx, input_yy, output_xx, output_yy = [],[],[],[]
    for i in range(n_batch):
        peaks = gen_XrXp()
        input_x, input_y, output_x, output_y = get_policy_ob_rollout(peaks, compl_model, n_pts, epi)
        input_xx.append(input_x)
        input_yy.append(input_y)
        output_xx.append(output_x)
        output_yy.append(output_y)

    return np.array(input_xx), np.array(input_yy), np.array(output_xx), np.array(output_yy)


def train_on_policy_ob():
    compl = Compl(100).cuda()

    MAX_RANGE = 100000
    for i in tqdm(range(MAX_RANGE)):
        epi = 1 - i / MAX_RANGE 
        n_obs = random.choice(list(range(1,20)))
        xx,yy,xx_new,yy_new = gen_batch_policy_ob_data(compl, n_obs, 100, epi)
        loss = compl.learn_once(xx, yy, xx_new, yy_new)

        if i % 100 == 0:
            print (f"------------------------------ epi {epi}")
            print ("number observations ", n_obs)
            print ("loss ", loss)
            mu, sig = compl.predict(xx,yy,xx_new)
            print ("mu: ", mu[0])
            print ("std: ", sig[0])
            print ("y*: ", yy_new[0])
            compl.save("./saved_models/ver1.mdl")

if __name__ == '__main__':
    train_on_policy_ob()
