import torch as th
from experiment1 import experiment1
from experiment2 import experiment2

if __name__ == "__main__":
    algo = ['C51', 'QR_DQN', 'IQN']
    # device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    device = 'cpu'
    print(device)

    # experiment1 : fitting ability
    # exp1 = experiment1()
    # exp1.train(algo[2])

    # experiment2 : learning in a simple game environment
    exp2 = experiment2(device)
    exp2.set_Algo(algo[2])
    exp2.train(3)
    exp2.vis()

    exp2.set_Env(True)
    exp2.set_Algo(algo[2])
    exp2.test()