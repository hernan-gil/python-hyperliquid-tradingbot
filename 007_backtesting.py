from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from ta.trend import ADXIndicator
import pandas as pd

class HiLoBreakOutStrategy(Strategy):
    adx_period=14
    adx_low=25
    adx_high=35

    def init(self):
        high = self.data.High
        low = self.data.Low
        close = self.data.Close

        price_data = pd.DataFrame({'High': high, 'Low':low, 'Close':close})

        self.adx = self.I(ADXIndicator(price_data['High'],price_data['Low'],price_data['Close'],self.adx_period).adx)

    def next(self):
        if self.adx[-1] > self.adx_low and self.adx[-1] < self.adx_high:
            if crossover(self.data.Close, self.data.High[-2]):
                self.sell()
            else:
                self.buy()

# Cargar los datos
data = pd.read_csv('../archive/btcusd_1-min_data_kaggle.csv', index_col='Timestamp')
#data['Date'] = pd.to_datetime(data['Date'])
#data.set_index('Date', inplace=True)

backtest = Backtest(data, HiLoBreakOutStrategy, cash=1000000, commission=0.002)
output = backtest.run()
print(output)
