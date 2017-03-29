import matplotlib.pyplot as plt
import numpy as np
import pdb

f=open("weight.txt","r")
reward_list = [[] for i in range(11)]
_temp=[]
xo=0
for count,f_l in enumerate(f):
    if count>6:
        if count%2==1:
            e=f_l.strip()[1:].split() 
            e = [float(_e) for _e in e]
            _temp.extend(e)
        else:
            e=f_l.strip()[:-1].split()
            e = [float(_e) for _e in e]
            _temp.extend(e)
    else:
        e=f_l.strip()[1:-2].split() #.split(" ").split("]").split("[")
        e = [float(_e) for _e in e]
        _temp=e
    #print(e)
#    print(len(reward_list[0]))
#    print(len(_temp))
#    print(len(reward_list[0]))
#    print("-----------")
    if len(_temp)==11:
        xo+=1
        if xo==1000:
            for idx,_e in enumerate(_temp):
                reward_list[idx].append(_e)
            xo=0
        _temp=[]
#pdb.set_trace()
del reward_list[9]
for idx,rew in enumerate(reward_list):
    plt.plot(rew[:2000],label="figure-"+str(idx+1))
plt.legend()
plt.savefig("weight.png")
#    e,r=f_l.strip().split("-")
#    e,r = int(e),int(r)
#    reward_list.append(r)
#print(len(reward_list))
#plt.plot(reward_list)
#plt.savefig("score3.png")
