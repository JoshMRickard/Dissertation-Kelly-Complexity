import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score, precision_score, recall_score, accuracy_score

class Backtest_tc():
    def __init__(self, **kwargs):
        self.z = 1000
        self.T = 12
        self.use_type = np.float32
        for key, value in kwargs.items():
            setattr(self, key, value)
    def predict(self, X: np.array, y: np.array):
        backtest = []
        T_max, self.P = X.shape
        self.c = self.P / self.T
        index = list(range(self.T, T_max))
        for t in index:
            S_train = X[t-self.T:t].astype(self.use_type)
            R_train = y[t-self.T:t].astype(self.use_type)
            S_test = X[t:t+1].astype(self.use_type)
            R_test = y[t:t+1].astype(self.use_type)
            beta = Ridge(alpha=(self.z*self.T), solver="svd", fit_intercept=False).fit(S_train, R_train).coef_
            forecast = S_test @ beta
            timing_strategy = forecast * R_test
            backtest.append({
                "index": R_test.index[0],
                "forecast": forecast[0],
                "timing_strategy": timing_strategy[0],
                "market_return": R_test[0]
            })
        backtest = pd.DataFrame(backtest)
        df = backtest
        ptc = float(0.001)
        df["delta"] = df["forecast"].diff()
        df["delta"][0] = df["forecast"][0]
        df["ptcdelta"] = df["delta"] * ptc
        df["timing_strategy"] = df["timing_strategy"] - df["ptcdelta"]
        backtest = df
        # The last value for market_return is NaN since it is predicting the next month
        self.backtest = pd.DataFrame(backtest).set_index("index")
        self.prediction = self.backtest["forecast"]
        return self

    def performance(self, time_factor:int = 12):
        data = self.backtest.dropna()
        market_reg = LinearRegression().fit(data[["market_return"]], data["timing_strategy"])
        beta = market_reg.coef_[0]
        alpha = market_reg.intercept_
        mean = data["timing_strategy"].mean()*time_factor
        std = data["timing_strategy"].std()*np.sqrt(time_factor)
        mean_market = data["market_return"].mean()*time_factor
        self.performance = {
            "Market Sharpe Ratio" : (data["market_return"].mean()*time_factor) / (data["market_return"].std()*np.sqrt(time_factor)),
            "Expected Return" : mean,
            "Volatility" : std,
            "R2" : r2_score(data["market_return"], data["forecast"]),
            "SR" : mean/std,
        }
        return self.performance

    
