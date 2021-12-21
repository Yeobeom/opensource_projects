# **서울과학술대학교 오픈소스프로그래밍**

## **프로젝트 개요**
LSTM(Long Short-Term Memory)를 이용하여 이더리움의 시간별 OHLCV(시가,최고가,최저가,종가,매매량)을 바탕으로 회귀분석(regression)을 통하여 이더리움의 가격을 예측한다.
  
## **요구 패키지**
* ccxt
* torch
* matplotlib
* sklearn
* numpy

# 사용법
ETH_LSTM.py 를 실행시, data

# 파일별 설명
## binance_data_load.py
binance API를 이용하여 이더리움의 OHLCV 데이터를 가져와 csv파일로 저장하거나, 데이터를 가져오는 클래스인 binance_data 가 존재하는 .py파일입니다.

## ETH_LSTM.py
LSTM를 이용하여 딥러닝을 하는 .py파일입니다.

* binance 관련 변수
```python
# 바이낸스의 API 키 입력(없을시 '', 대신 최신 데이터 사용불가)
api_key = ''

# 바이낸스의 시크릿 키 입력(없을시 '', 대신 최신 데이터 사용불가)
secret = ''

# OHLCV의 시간 간격 (ex. '15m' -> 10시 15분 데이터, 10시 30분 데이터...)
t_frame = '1d' # '1m','15m','30m','1h','3h','1d'

# 가져올 코인 '코인/단위 임. 
symbol = 'ETH/USDT'
```
* LSTM 관련 변수
LSTM 관련 변수들로 적절히 변경 가능
```python
EPOCH_MAX = 50 # epoch 횟수
EPOCH_LOG = 1 # epoch 횟수별 표시 간격
OPTIMIZER_PARAM = {'lr': 0.01} # learning rate
SCHEDULER_PARAM = {'step_size': 5, 'gamma': 0.5} # scheduler param
USE_CUDA = torch.cuda.is_available() # CUDA 사용유무
SAVE_MODEL = f'./data/ETH_LSTM._{t_frame}pt' # Make empty('') if you don't want save the model
RANDOM_SEED = 777 # 랜덤시드 고정
DATA_LOADER_PARAM = {'batch_size': 50, 'shuffle': False} # data loader param
```

* 내용 설명
```python
class ETH_LSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super(ETH_LSTM, self).__init__()
        self.LSTM1 = torch.nn.LSTM(input_size, 128)
        self.LSTM2 = torch.nn.LSTM(128,128)
        self.fc = torch.nn.Linear(128, output_size)

    def forward(self, x):
        output, hidden = self.LSTM1(x)
        output, hidden = self.LSTM2(output)
        x = self.fc(output) # Use output of the last sequence
        return x
```

##결론
