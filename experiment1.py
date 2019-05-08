from data import *
from model2 import *

vae = VAE(162, 1300).cuda()
vae.load('vae1.mdl')

ucb = UCB(vae)

ro_lengths = []
print ("queries / rewards took to find max for each test rows ")
for test_row in M_test:
    env = Env(test_row)
    ro = get_rollout(env, ucb)
    print ([(r[1],r[2]) for r in ro])
    ro_lengths.append(len(ro))

print ("average tries until finding max ")
print (sum(ro_lengths) / len(ro_lengths))
to_save = sum(ro_lengths) / len(ro_lengths)

import datetime
time_stamp = "".join(str(datetime.datetime.now()).split())
result_path = "results/avgtime_+"time_stamp+".p"
import pickle
pickle.dump(to_save, open(result_path, "wb"))

# save the results
from scripts.launch import upload
import os
upload_script = upload(result_path)
os.system(upload_script)

