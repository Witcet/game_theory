import numpy as np


# 定义古诺模型类
class CournotCompetitionGame:
    def __init__(self, number_player):

        self.number_player = number_player  # 参与人数
        self.history_quantity = np.empty([number_player])  # 产量历史纪录 1x2
        self.history_costs = np.empty([number_player])  # 成本历史记录 1x2
        self.history_price = np.empty([1, 1])  # 产品价格
        self.history_profits = np.empty([number_player])  # 利润历史记录 1x2
        self.expected_shape = self.history_quantity.shape  # （1，）
        self.first_entry = True

        # 纳什均衡：均为(a-c)/(n+1)b
        self.max_price = 150  # a
        self.quantity_slope = 1  # b
        self.marginal_cost = 30  # c

    def get_equilibrium_production(self):
        num = self.number_player + 1
        q = (self.max_price - self.marginal_cost) / (num * self.quantity_slope)  # 产量理论计算公式
        return q

    def get_all_history(self):
        if len(self.history_price) > 1:
            out = np.hstack([self.history_quantity])
        else:
            out = np.hstack([self.history_quantity.reshape(1, self.number_player)])
        return out

    def get_lag_history(self, lag=1):
        if len(self.history_price) > 1:
            lag_history = np.append(self.get_all_history()[-lag, :],
                                    self.get_all_history()[-(lag + 1), :])
        else:
            lag_history = np.append(self.get_all_history()[-lag, :],
                                    self.get_all_history()[-lag, :])
        return lag_history

    def get_price(self, produced_quantities):
        # p = a-b*Q
        price = self.max_price - self.quantity_slope * np.sum(produced_quantities)
        return price

    def get_costs(self, produced_quantities):
        # Ci = Qi*c
        costs = produced_quantities * self.marginal_cost
        return costs

    def get_profits(self, produced_quantities, consult_only=False):
        costs = self.get_costs(produced_quantities)
        price = self.get_price(produced_quantities)

        # Si = p*Qi-Ci = (p-c)*Qi
        profits = price * produced_quantities - costs

        if not consult_only:
            self.history_quantity = np.vstack([self.history_quantity, produced_quantities])  # n*2
            self.history_profits = np.vstack([self.history_profits, profits])  # n*2
            self.history_costs = np.vstack([self.history_costs, costs])  # n*2
            self.history_price = np.append(self.history_price, price)  # n*1
            if self.first_entry:
                self.first_entry = False
                self.history_quantity = self.history_quantity[1:, ]
                self.history_profits = self.history_profits[1:, ]
                self.history_costs = self.history_costs[1:, ]
                self.history_price = self.history_price[1:]

        return profits
