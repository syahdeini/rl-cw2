import matplotlib.pyplot as plt
import numpy as np

f=open("score3.txt","r")
reward_list = []
for f_l in f:
    e,r=f_l.strip().split("-")
    e,r = int(e),int(r)
    reward_list.append(r)
print(len(reward_list))
plt.plot(reward_list)
plt.savefig("score3.png")
