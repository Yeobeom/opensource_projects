import ccxt
import pandas as pd
import datetime
import os
import time
import numpy as np

class binance_data():
    now = datetime.datetime.now()
    timestamp_now = int(time.time()*1000)
    addtime = {'1m':60000, '15m':900000, '30m':1800000,'1h':3600000, '12h':43200000,'1d':86400000}
    def __init__(self,api_key,secret,ticker,t_frame):
        self.ticker = ticker
        self.t_frame = t_frame
        self.file = f'./data/binance_ETH_{self.t_frame}.csv'
        if api_key != '':
            self.binance = ccxt.binance(config={
                'apiKey': api_key, 
                'secret': secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future'
                }
            })
        else:
            pass
        
    def updating_coin_csv(self):
        ##ethsince = 1577059200000
        if not os.path.isfile(self.file):
            if not os.path.isdir('./data'):
                os.mkdir('data')
            ethsince = 1577059200000
        else: # else it exists so append without writing the header
            ethsince = int(pd.read_csv(self.file).iloc[-1]["Timestamp"])+self.addtime[self.t_frame]

        while ethsince < self.timestamp_now:
            print("Updating Dataset...")
            eth_ohlcv = self.binance.fetch_ohlcv(self.ticker,timeframe=self.t_frame, since=ethsince ,limit=1000)
            df = pd.DataFrame(eth_ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
            df['Timestamp'] = df['datetime']
            df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
            df.set_index('datetime', inplace=True)
            if not os.path.isfile(self.file):
                df.to_csv(self.file, header='column_names')
            else: # else it exists so append without writing the header
                df.to_csv(self.file, mode='a', header=False)
            ethsince = int(pd.read_csv(self.file).iloc[-1]["Timestamp"])+self.addtime[self.t_frame]
        print("Dataset Updated!")

    def Load_Coin_Data(self):
        class CoinData:
            data = []
            targets = []
            target_names = ['predicted_open','predicted_close']
            features = []
            features_names = ['Timestamp']
            recent_data = []
        coindata = CoinData
        df = pd.read_csv(self.file)
        df.loc[:,'predicted_close'] = pd.Series(df.loc[1:,'close'].to_list())
        df.loc[:,'predicted_open'] = pd.Series(df.loc[1:,'open'].to_list())
        coindata.recent_data = df.iloc[-1].to_list()
        df = df.drop(df.index[-1])
        print('Converting Data...')
        # using Datafrem.apply function fill the data and targets
        df.apply(lambda row: coindata.data.append(row[['open','high','low','close','volume']].to_list()), axis=1)
        df.apply(lambda row: coindata.targets.append(row[['predicted_open','predicted_close']].to_list()), axis=1)
        df.apply(lambda row: coindata.features.append([row['Timestamp']]), axis=1)

        coindata.data = np.array(coindata.data)
        coindata.targets = np.array(coindata.targets)
        coindata.features = np.array(coindata.features)
        print('Data Converted!')
        return coindata
    