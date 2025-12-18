from experiment1 import experiment1
from experiment2 import experiment2

if __name__ == "__main__":
    algo = ['C51', 'QR_DQN', 'IQN']
    device = 'cpu'

    # experiment1 : fitting ability
    exp1 = experiment1()
    exp1.train(algo[2])

    # experiment2 : learning in a simple game environment (MountainCar)
    # train
    exp2 = experiment2(device, algo[0])
    exp2.train(3)
    exp2.vis()
    # test
    exp2.set_Env(True)
    exp2.set_Algo()
    exp2.test()