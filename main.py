from game import CournotCompetitionGame
import numpy as np
import random
from method import PlayerRndForrest
from method import PlayerNNet
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['KaiTi']

# 初始产量设置
initial_quantity = [92.12, 25.68]
# initial_quantity = [65.39, 73.15]
# initial_quantity = [10.45, 41.29, 88.09]

nb_player = 2  # 参与竞争的企业个数
game = CournotCompetitionGame(nb_player)
players = []
for i in range(nb_player):
    # 选用随机森林还是神经网络，前者正确率高，后者速度较快
    pl = PlayerRndForrest(game=game, end_of_random_action_time=1500, action_range=10, random_action_range=1)
    # pl.production = random.random() * 40 + 10  # 随机初始值
    pl.production = initial_quantity[i]
    players.append(pl)
    del pl

S = 2000  # 迭代次数
for s in range(S):
    state = game.get_lag_history()
    for pl in players:
        pl_action = pl.select_action(current_state=state, step=s)
        pl.production = max(0, pl.production + pl_action)

    production_list = []
    for temp_pl in players:
        production_list.append(temp_pl.production)
    production_vector = np.array(production_list)
    del production_list
    profits = game.get_profits(produced_quantities=production_vector)
    next_state = game.get_lag_history()
    for j in range(len(players)):
        pl = players[j]
        pl.update_q(state=state, next_state=next_state, reward=profits[j])

    if s % 100 == 0:
        print('---- 当前进度:', '{:.2%}'.format(s / S), '----')

# history_quantity转成一行的array
game.get_all_history()
game.get_lag_history()
# history_quantity转list
temp = np.array(game.history_quantity.tolist())
plt.plot(game.history_quantity)
plt.title("产量")
plt.ylim((0, 120))
plt.show()

# plt.plot(game.history_profits)
# plt.title("利润")
# plt.show()
#
# plt.plot(np.sum(game.history_quantity, 1))
# plt.title("总产量")
# plt.show()
#
# plt.plot(np.sum(game.history_profits, 1))
# plt.title("总利润")
# plt.show()

ep = game.get_equilibrium_production()
error1 = (game.history_quantity[-1] - ep) / ep
error1 = np.round(error1, 4).tolist()
fp = np.round(game.history_quantity[-1], 2).tolist()
average = np.mean(game.history_quantity[-250:], axis=0)  # 取最后250次的结果取平均
error2 = (average - ep) / ep
average = np.round(average, 2).tolist()
error2 = np.round(error2, 4).tolist()

print("理论上达成纳什均衡时的产量： ", ep)
print("初始产量：  ", initial_quantity)
print("最终产量：  ", fp)
print("相对误差：  ", error1)
print("平均产量：  ", average)
print("相对误差：  ", error2)
