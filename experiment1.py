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
