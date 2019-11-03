#!/usr/bin/python
import csv
import time
import json
import talib
import requests
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta, timezone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE

headers = {'Content-Type': 'application/json'}
api_url_base = 'https://public.bitbank.cc'
pair = 'btc_jpy'
period = '1min'

today = datetime.today()
yesterday = today - timedelta(days=1)
today = "{0:%Y%m%d}".format(today)
yesterday = "{0:%Y%m%d}".format(yesterday)

def api_ohlcv(timestamp):
    api_url = '{0}/{1}/candlestick/{2}/{3}'.format(api_url_base, pair, period, timestamp)
    response = requests.get(api_url, headers=headers)

    if response.status_code == 200:
        ohlcv = json.loads(response.content.decode('utf-8'))['data']['candlestick'][0]['ohlcv']
        return ohlcv
    else:
        return None

def main():
    ohlcv = api_ohlcv('20191017')
    open, high, low, close, volume, timestamp = [],[],[],[],[],[]

    for i in ohlcv:
        open.append(int(i[0]))
        high.append(int(i[1]))
        low.append(int(i[2]))
        close.append(int(i[3]))
        volume.append(float(i[4]))
        time_str = str(i[5])
        timestamp.append(datetime.fromtimestamp(int(time_str[:10])).strftime('%Y/%m/%d %H:%M:%M'))

    date_time_index = pd.to_datetime(timestamp) # convert to DateTimeIndex type
    df = pd.DataFrame({'open': open, 'high': high, 'low': low, 'close': close, 'volume': volume}, index=date_time_index)
    # df.index += pd.offsets.Hour(9) # adjustment for JST if required
    print(df.shape)
    print(df.columns)

    # pct_change
    f = lambda x: 1 if x>0.0001 else -1 if x<-0.0001 else 0 if -0.0001<=x<=0.0001 else np.nan
    y = df.rename(columns={'close': 'y'}).loc[:, 'y'].pct_change(1).shift(-1).fillna(0)
    X = df.copy()
    y_ = pd.DataFrame(y.map(f), columns=['y'])
    y = df.rename(columns={'close': 'y'}).loc[:, 'y'].pct_change(1).fillna(0)
    df_ = pd.concat([X, y_], axis=1)

    # check the shape
    print('----------------------------------------------------------------------------------------')
    print('X shape: (%i,%i)' % X.shape)
    print('y shape: (%i,%i)' % y_.shape)
    print('----------------------------------------------------------------------------------------')
    print(y_.groupby('y').size())
    print('y=1 up, y=0 stay, y=-1 down')
    print('----------------------------------------------------------------------------------------')

    # feature calculation
    open = pd.Series(df['open'])
    high = pd.Series(df['high'])
    low = pd.Series(df['low'])
    close = pd.Series(df['close'])
    volume = pd.Series(df['volume'])

    # pct_change for new column
    X['diff'] = y

    # Exponential Moving Average
    ema = talib.EMA(close, timeperiod=3)
    ema = ema.fillna(ema.mean())

    # Momentum
    momentum = talib.MOM(close, timeperiod=5)
    momentum = momentum.fillna(momentum.mean())

    # RSI
    rsi = talib.RSI(close, timeperiod=14)
    rsi = rsi.fillna(rsi.mean())

    # ADX
    adx = talib.ADX(high, low, close, timeperiod=14)
    adx = adx.fillna(adx.mean())

    # ADX change
    adx_change = adx.pct_change(1).shift(-1)
    adx_change = adx_change.fillna(adx_change.mean())

    # AD
    ad = talib.AD(high, low, close, volume)
    ad = ad.fillna(ad.mean())

    X_ = pd.concat([X, ema, momentum, rsi, adx_change, ad], axis=1).drop(['open', 'high', 'low', 'close'], axis=1)
    X_.columns = ['volume','diff', 'ema', 'momentum', 'rsi', 'adx', 'ad']
    X_.join(y_).head(10)

    # default parameter models
    X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.33, random_state=42)
    print('X_train shape: {}'.format(X_train.shape))
    print('X_test shape: {}'.format(X_test.shape))
    print('y_train shape: {}'.format(y_train.shape))
    print('y_test shape: {}'.format(y_test.shape))

    pipe_knn = Pipeline([('scl', StandardScaler()), ('est', KNeighborsClassifier(n_neighbors=3))])
    pipe_logistic = Pipeline([('scl', StandardScaler()), ('est', LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=39))])
    pipe_rf = Pipeline([('scl', StandardScaler()), ('est', RandomForestClassifier(random_state=39))])
    pipe_gb = Pipeline([('scl', StandardScaler()), ('est', GradientBoostingClassifier(random_state=39))])

    pipe_names = ['KNN','Logistic','RandomForest','GradientBoosting']
    pipe_lines = [pipe_knn, pipe_logistic, pipe_rf, pipe_gb]

    for (i, pipe) in enumerate(pipe_lines):
        pipe.fit(X_train, y_train.values.ravel())
        print(pipe)
        print('%s: %.3f' % (pipe_names[i] + ' Train Accuracy', accuracy_score(y_train.values.ravel(), pipe.predict(X_train))))
        print('%s: %.3f' % (pipe_names[i] + ' Test Accuracy', accuracy_score(y_test.values.ravel(), pipe.predict(X_test))))
        print('%s: %.3f' % (pipe_names[i] + ' Train F1 Score', f1_score(y_train.values.ravel(), pipe.predict(X_train), average='micro')))
        print('%s: %.3f' % (pipe_names[i] + ' Test F1 Score', f1_score(y_test.values.ravel(), pipe.predict(X_test), average='micro')))

    for (i, pipe) in enumerate(pipe_lines):
        predict = pipe.predict(X_test)
        cm = confusion_matrix(y_test.values.ravel(), predict, labels=[-1, 0, 1])
        print('{} Confusion Matrix'.format(pipe_names[i]))
        print(cm)

    ## Overlap Studies Functions
    # DEMA - Double Exponential Moving Average
    dema = talib.DEMA(close, timeperiod=3)
    dema = dema.fillna(dema.mean())
    print('DEMA - Double Exponential Moving Average shape: {}'.format(dema.shape))

    # EMA - Exponential Moving Average
    ema = talib.EMA(close, timeperiod=3)
    ema = ema.fillna(ema.mean())
    print('EMA - Exponential Moving Average shape: {}'.format(ema.shape))

    # HT_TRENDLINE - Hilbert Transform - Instantaneous Trendline
    hilbert = talib.HT_TRENDLINE(close)
    hilbert = hilbert.fillna(hilbert.mean())
    print('HT_TRENDLINE - Hilbert Transform - Instantaneous Trendline shape: {}'.format(hilbert.shape))

    # KAMA - Kaufman Adaptive Moving Average
    kama = talib.KAMA(close, timeperiod=3)
    kama = kama.fillna(kama.mean())
    print('KAMA - Kaufman Adaptive Moving Average shape: {}'.format(kama.shape))

    # MA - Moving average
    ma = talib.MA(close, timeperiod=3, matype=0)
    ma = ma.fillna(ma.mean())
    print('MA - Moving average shape: {}'.format(kama.shape))

    # MIDPOINT - MidPoint over period
    midpoint = talib.MIDPOINT(close, timeperiod=7)
    midpoint = midpoint.fillna(midpoint.mean())
    print('MIDPOINT - MidPoint over period shape: {}'.format(midpoint.shape))

    # MIDPRICE - Midpoint Price over period
    midprice = talib.MIDPRICE(high, low, timeperiod=7)
    midprice = midprice.fillna(midprice.mean())
    print('MIDPRICE - Midpoint Price over period shape: {}'.format(midprice.shape))

    # SAR - Parabolic SAR
    sar = talib.SAR(high, low, acceleration=0, maximum=0)
    sar = sar.fillna(sar.mean())
    print('SAR - Parabolic SAR shape: {}'.format(sar.shape))

    # SAREXT - Parabolic SAR - Extended
    sarext = talib.SAREXT(high, low, startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)
    sarext = sarext.fillna(sarext.mean())
    print('SAREXT - Parabolic SAR - Extended shape: {}'.format(sarext.shape))

    # SMA - Simple Moving Average
    sma = talib.SMA(close, timeperiod=3)
    sma = sma.fillna(sma.mean())
    print('SMA - Simple Moving Average shape: {}'.format(sma.shape))

    # T3 - Triple Exponential Moving Average (T3)
    t3 = talib.T3(close, timeperiod=5, vfactor=0)
    t3 = t3.fillna(t3.mean())
    print('T3 - Triple Exponential Moving Average shape: {}'.format(t3.shape))

    # TEMA - Triple Exponential Moving Average
    tema = talib.TEMA(close, timeperiod=3)
    tema = tema.fillna(tema.mean())
    print('TEMA - Triple Exponential Moving Average shape: {}'.format(tema.shape))

    # TRIMA - Triangular Moving Average
    trima = talib.TRIMA(close, timeperiod=3)
    trima = trima.fillna(trima.mean())
    print('TRIMA - Triangular Moving Average shape: {}'.format(trima.shape))

    # WMA - Weighted Moving Average
    wma = talib.WMA(close, timeperiod=3)
    wma = wma.fillna(wma.mean())
    print('WMA - Weighted Moving Average shape: {}'.format(wma.shape))

    ## Momentum Indicator Functions
    # ADX - Average Directional Movement Index
    adx = talib.ADX(high, low, close, timeperiod=14)
    adx = adx.fillna(adx.mean())
    print('ADX - Average Directional Movement Index shape: {}'.format(adx.shape))

    # ADXR - Average Directional Movement Index Rating
    adxr = talib.ADXR(high, low, close, timeperiod=7)
    adxr = adxr.fillna(adxr.mean())
    print('ADXR - Average Directional Movement Index Rating shape: {}'.format(adxr.shape))

    # APO - Absolute Price Oscillator
    apo = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
    apo = apo.fillna(apo.mean())
    print('APO - Absolute Price Oscillator shape: {}'.format(apo.shape))

    # AROONOSC - Aroon Oscillator
    aroon = talib.AROONOSC(high, low, timeperiod=14)
    aroon = aroon.fillna(aroon.mean())
    print('AROONOSC - Aroon Oscillator shape: {}'.format(apo.shape))

    # BOP - Balance Of Power
    bop = talib.BOP(open, high, low, close)
    bop = bop.fillna(bop.mean())
    print('BOP - Balance Of Power shape: {}'.format(apo.shape))

    # CCI - Commodity Channel Index
    cci = talib.CCI(high, low, close, timeperiod=7)
    cci = cci.fillna(cci.mean())
    print('CCI - Commodity Channel Index shape: {}'.format(cci.shape))

    # CMO - Chande Momentum Oscillator
    cmo = talib.CMO(close, timeperiod=7)
    cmo = cmo.fillna(cmo.mean())
    print('CMO - Chande Momentum Oscillator shape: {}'.format(cmo.shape))

    # DX - Directional Movement Index
    dx = talib.DX(high, low, close, timeperiod=7)
    dx = dx.fillna(dx.mean())
    print('DX - Directional Movement Index shape: {}'.format(dx.shape))

    # MFI - Money Flow Index
    mfi = talib.MFI(high, low, close, volume, timeperiod=7)
    mfi = mfi.fillna(mfi.mean())
    print('MFI - Money Flow Index shape: {}'.format(mfi.shape))

    # MINUS_DI - Minus Directional Indicator
    minusdi = talib.MINUS_DI(high, low, close, timeperiod=14)
    minusdi = minusdi.fillna(minusdi.mean())
    print('MINUS_DI - Minus Directional Indicator shape: {}'.format(minusdi.shape))

    # MINUS_DM - Minus Directional Movement
    minusdm = talib.MINUS_DM(high, low, timeperiod=14)
    minusdm = minusdm.fillna(minusdm.mean())
    print('MINUS_DM - Minus Directional Movement shape: {}'.format(minusdm.shape))

    # MOM - Momentum
    mom = talib.MOM(close, timeperiod=5)
    mom = mom.fillna(mom.mean())
    print('MOM - Momentum shape: {}'.format(mom.shape))

    # PLUS_DI - Plus Directional Indicator
    plusdi = talib.PLUS_DI(high, low, close, timeperiod=14)
    plusdi = plusdi.fillna(plusdi.mean())
    print('PLUS_DI - Plus Directional Indicator shape: {}'.format(plusdi.shape))

    # PLUS_DM - Plus Directional Movement
    plusdm = talib.PLUS_DM(high, low, timeperiod=14)
    plusdm = plusdm.fillna(plusdm.mean())
    print('PLUS_DM - Plus Directional Movement shape: {}'.format(plusdi.shape))

    # PPO - Percentage Price Oscillator
    ppo = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)
    ppo = ppo.fillna(ppo.mean())
    print('PPO - Percentage Price Oscillator shape: {}'.format(ppo.shape))

    # ROC - Rate of change:((price/prevPrice)-1)*100
    roc = talib.ROC(close, timeperiod=10)
    roc = roc.fillna(roc.mean())
    print('ROC - Rate of change : ((price/prevPrice)-1)*100 shape: {}'.format(roc.shape))

    # RSI - Relative Strength Index
    rsi = talib.RSI(close, timeperiod=14)
    rsi = rsi.fillna(rsi.mean())
    print('RSI - Relative Strength Index shape: {}'.format(rsi.shape))

    # TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
    trix = talib.TRIX(close, timeperiod=30)
    trix = trix.fillna(trix.mean())
    print('TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA shape: {}'.format(trix.shape))

    # ULTOSC - Ultimate Oscillator
    ultosc = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    ultosc = ultosc.fillna(ultosc.mean())
    print('ULTOSC - Ultimate Oscillator shape: {}'.format(ultosc.shape))

    # WILLR - Williams'%R
    willr = talib.WILLR(high, low, close, timeperiod=7)
    willr = willr.fillna(willr.mean())
    print("WILLR - Williams'%R shape: {}".format(willr.shape))

    ## Volume Indicator Functions
    # AD - Chaikin A/D Line
    ad = talib.AD(high, low, close, volume)
    ad = ad.fillna(ad.mean())
    print('AD - Chaikin A/D Line shape: {}'.format(ad.shape))
      
    # ADOSC - Chaikin A/D Oscillator
    adosc = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    adosc = adosc.fillna(adosc.mean())
    print('ADOSC - Chaikin A/D Oscillator shape: {}'.format(adosc.shape))

    # OBV - On Balance Volume
    obv = talib.OBV(close, volume)
    obv = obv.fillna(obv.mean())
    print('OBV - On Balance Volume shape: {}'.format(obv.shape))

    ## Volatility Indicator Functions
    # ATR - Average True Range
    atr = talib.ATR(high, low, close, timeperiod=7)
    atr = atr.fillna(atr.mean())
    print('ATR - Average True Range shape: {}'.format(atr.shape))

    # NATR - Normalized Average True Range
    natr = talib.NATR(high, low, close, timeperiod=7)
    natr = natr.fillna(natr.mean())
    print('NATR - Normalized Average True Range shape: {}'.format(natr.shape))

    # TRANGE - True Range
    trange = talib.TRANGE(high, low, close)
    trange = trange.fillna(trange.mean())
    print('TRANGE - True Range shape: {}'.format(natr.shape))

    ## Price Transform Functions
    # AVGPRICE - Average Price
    avg = talib.AVGPRICE(open, high, low, close)
    avg = avg.fillna(avg.mean())
    print('AVGPRICE - Average Price shape: {}'.format(natr.shape))

    # MEDPRICE - Median Price
    medprice = talib.MEDPRICE(high, low)
    medprice = medprice.fillna(medprice.mean())
    print('MEDPRICE - Median Price shape: {}'.format(medprice.shape))

    # TYPPRICE - Typical Price
    typ = talib.TYPPRICE(high, low, close)
    typ = typ.fillna(typ.mean())
    print('TYPPRICE - Typical Price shape: {}'.format(typ.shape))

    # WCLPRICE - Weighted Close Price
    wcl = talib.WCLPRICE(high, low, close)
    wcl = wcl.fillna(wcl.mean())
    print('WCLPRICE - Weighted Close Price shape: {}'.format(wcl.shape))

    ## Cycle Indicator Functions
    # HT_DCPERIOD - Hilbert Transform - Dominant Cycle Period
    dcperiod = talib.HT_DCPERIOD(close)
    dcperiod = dcperiod.fillna(dcperiod.mean())
    print('HT_DCPERIOD - Hilbert Transform - Dominant Cycle Period shape: {}'.format(dcperiod.shape))

    # HT_DCPHASE - Hilbert Transform - Dominant Cycle Phase
    dcphase = talib.HT_DCPHASE(close)
    dcphase = dcphase.fillna(dcphase.mean())
    print('HT_DCPHASE - Hilbert Transform - Dominant Cycle Phase shape: {}'.format(dcperiod.shape))

    ## Statistic Functions
    # BETA - Beta
    beta = talib.BETA(high, low, timeperiod=3)
    beta = beta.fillna(beta.mean())
    print('BETA - Beta shape: {}'.format(beta.shape))

    # CORREL - Pearson's Correlation Coefficient(r)
    correl = talib.CORREL(high, low, timeperiod=30)
    correl = correl.fillna(correl.mean())
    print("CORREL - Pearson's Correlation Coefficient(r) shape: {}".format(beta.shape))

    # LINEARREG - Linear Regression
    linreg = talib.LINEARREG(close, timeperiod=7)
    linreg = linreg.fillna(linreg.mean())
    print("LINEARREG - Linear Regression shape: {}".format(linreg.shape))

    # STDDEV - Standard Deviation
    stddev = talib.STDDEV(close, timeperiod=5, nbdev=1)
    stddev = stddev.fillna(stddev.mean())
    print("STDDEV - Standard Deviation shape: {}".format(stddev.shape))

    # TSF - Time Series Forecast
    tsf = talib.TSF(close, timeperiod=7)
    tsf = tsf.fillna(tsf.mean())
    print("TSF - Time Series Forecast shape: {}".format(tsf.shape))

    # VAR - Variance
    var = talib.VAR(close, timeperiod=5, nbdev=1)
    var = var.fillna(var.mean())
    print("VAR - Variance shape: {}".format(var.shape))

    ## Feature DataFrame
    X_full = pd.concat([X, dema, ema, hilbert, kama, ma, midpoint, midprice, sar, sarext, sma, t3, tema, trima, wma, adx, adxr, apo, aroon, bop, cci, cmo, mfi, minusdi, minusdm, mom, plusdi, plusdm, ppo, roc, rsi, trix, ultosc, willr, ad, adosc, obv, atr, natr, trange, avg, medprice, typ, wcl, dcperiod, dcphase, beta, correl, linreg, stddev, tsf, var], axis=1).drop(['open', 'high', 'low', 'close'], axis=1)
    X_full.columns = ['volume','diff', 'dema', 'ema', 'hilbert', 'kama', 'ma', 'midpoint', 'midprice', 'sar', 'sarext', 'sma', 't3', 'tema', 'trima', 'wma', 'adx', 'adxr', 'apo', 'aroon', 'bop', 'cci', 'cmo', 'mfi', 'minusdi', 'minusdm', 'mom', 'plusdi', 'plusdm', 'ppo', 'roc', 'rsi', 'trix', 'ultosc', 'willr', 'ad', 'adosc', 'obv', 'atr', 'natr', 'trange', 'avg', 'medprice', 'typ', 'wcl', 'dcperiod', 'dcphase', 'beta', 'correl', 'linreg', 'stddev', 'tsf', 'var']
    X_full.join(y_).head(10)

    # full feature models
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_full, y_, test_size=0.33, random_state=42)
    print('X_train shape: {}'.format(X_train_full.shape))
    print('X_test shape: {}'.format(X_test_full.shape))
    print('y_train shape: {}'.format(y_train_full.shape))
    print('y_test shape: {}'.format(y_test_full.shape))

    pipe_knn_full = Pipeline([('scl', StandardScaler()), ('est', KNeighborsClassifier(n_neighbors=3))])
    pipe_logistic_full = Pipeline([('scl', StandardScaler()), ('est', LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=39))])
    pipe_rf_full = Pipeline([('scl', StandardScaler()), ('est', RandomForestClassifier(random_state=39))])
    pipe_gb_full = Pipeline([('scl', StandardScaler()), ('est', GradientBoostingClassifier(random_state=39))])

    pipe_names = ['KNN','Logistic','RandomForest','GradientBoosting']
    pipe_lines_full = [pipe_knn_full, pipe_logistic_full, pipe_rf_full, pipe_gb_full]

    for (i, pipe) in enumerate(pipe_lines_full):
        pipe.fit(X_train_full, y_train_full.values.ravel())
        print(pipe)
        print('%s: %.3f' % (pipe_names[i] + ' Train Accuracy', accuracy_score(y_train_full.values.ravel(), pipe.predict(X_train_full))))
        print('%s: %.3f' % (pipe_names[i] + ' Test Accuracy', accuracy_score(y_test_full.values.ravel(), pipe.predict(X_test_full))))
        print('%s: %.3f' % (pipe_names[i] + ' Train F1 Score', f1_score(y_train_full.values.ravel(), pipe.predict(X_train_full), average='micro')))
        print('%s: %.3f' % (pipe_names[i] + ' Test F1 Score', f1_score(y_test_full.values.ravel(), pipe.predict(X_test_full), average='micro')))

    # Univariate Statistics
    select = SelectPercentile(percentile=25)
    select.fit(X_train_full, y_train_full.values.ravel())
    X_train_selected = select.transform(X_train_full)
    X_test_selected = select.transform(X_test_full)
    # GradientBoost Classifier
    print('--------------------------Without Univariate Statistics-------------------------------------')
    pipe_gb = Pipeline([('scl', StandardScaler()), ('est', GradientBoostingClassifier(random_state=39))])
    pipe_gb.fit(X_train_full, y_train_full.values.ravel())
    print('Train Accuracy: {:.3f}'.format(accuracy_score(y_train_full.values.ravel(), pipe_gb.predict(X_train_full))))
    print('Test Accuracy: {:.3f}'.format(accuracy_score(y_test_full.values.ravel(), pipe_gb.predict(X_test_full))))
    print('Train F1 Score: {:.3f}'.format(f1_score(y_train_full.values.ravel(), pipe_gb.predict(X_train_full), average='micro')))
    print('Test F1 Score: {:.3f}'.format(f1_score(y_test_full.values.ravel(), pipe_gb.predict(X_test_full), average='micro')))
    # GradientBoost Cllassifier with Univariate Statistics
    print('---------------------------With Univariate Statistics--------------------------------------')
    pipe_gb_percentile = Pipeline([('scl', StandardScaler()), ('est', GradientBoostingClassifier(random_state=39))])
    pipe_gb_percentile.fit(X_train_selected, y_train_full.values.ravel())
    print('Train Accuracy: {:.3f}'.format(accuracy_score(y_train_full.values.ravel(), pipe_gb_percentile.predict(X_train_selected))))
    print('Test Accuracy: {:.3f}'.format(accuracy_score(y_test_full.values.ravel(), pipe_gb_percentile.predict(X_test_selected))))
    print('Train F1 Score: {:.3f}'.format(f1_score(y_train_full.values.ravel(), pipe_gb_percentile.predict(X_train_selected), average='micro')))
    print('Test F1 Score: {:.3f}'.format(f1_score(y_test_full.values.ravel(), pipe_gb_percentile.predict(X_test_selected), average='micro')))

    # Model-based Selection
    select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold="1.25*mean")
    select.fit(X_train_full, y_train_full.values.ravel())
    X_train_model = select.transform(X_train_full)
    X_test_model = select.transform(X_test_full)
    # GradientBoost Classifier
    print('--------------------------Without Model-based Selection--------------------------------------')
    pipe_gb = Pipeline([('scl', StandardScaler()), ('est', GradientBoostingClassifier(random_state=39))])
    pipe_gb.fit(X_train_full, y_train_full.values.ravel())
    print('Train Accuracy: {:.3f}'.format(accuracy_score(y_train_full.values.ravel(), pipe_gb.predict(X_train_full))))
    print('Test Accuracy: {:.3f}'.format(accuracy_score(y_test_full.values.ravel(), pipe_gb.predict(X_test_full))))
    print('Train F1 Score: {:.3f}'.format(f1_score(y_train_full.values.ravel(), pipe_gb.predict(X_train_full), average='micro')))
    print('Test F1 Score: {:.3f}'.format(f1_score(y_test_full.values.ravel(), pipe_gb.predict(X_test_full), average='micro')))
    # GradientBoost Classifier with Model-based Selection
    print('----------------------------With Model-based Selection--------------------------------------')
    pipe_gb_model = Pipeline([('scl', StandardScaler()), ('est', GradientBoostingClassifier(random_state=39))])
    pipe_gb_model.fit(X_train_model, y_train_full.values.ravel())
    print('Train Accuracy: {:.3f}'.format(accuracy_score(y_train_full.values.ravel(), pipe_gb_model.predict(X_train_model))))
    print('Test Accuracy: {:.3f}'.format(accuracy_score(y_test_full.values.ravel(), pipe_gb_model.predict(X_test_model))))
    print('Train F1 Score: {:.3f}'.format(f1_score(y_train_full.values.ravel(), pipe_gb_model.predict(X_train_model), average='micro')))
    print('Test F1 Score: {:.3f}'.format(f1_score(y_test_full.values.ravel(), pipe_gb_model.predict(X_test_model), average='micro')))

    # Recursive Feature Elimination
    select = RFE(RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=15)
    select.fit(X_train_full, y_train_full.values.ravel())
    X_train_rfe = select.transform(X_train_full)
    X_test_rfe = select.transform(X_test_full)
    # GradientBoost Classifier
    print('--------------------------Without Recursive Feature Elimination-------------------------------------')
    pipe_gb = Pipeline([('scl', StandardScaler()), ('est', GradientBoostingClassifier(random_state=39))])
    pipe_gb.fit(X_train_full, y_train_full.values.ravel())
    print('Train Accuracy: {:.3f}'.format(accuracy_score(y_train_full.values.ravel(), pipe_gb.predict(X_train_full))))
    print('Test Accuracy: {:.3f}'.format(accuracy_score(y_test_full.values.ravel(), pipe_gb.predict(X_test_full))))
    print('Train F1 Score: {:.3f}'.format(f1_score(y_train_full.values.ravel(), pipe_gb.predict(X_train_full), average='micro')))
    print('Test F1 Score: {:.3f}'.format(f1_score(y_test_full.values.ravel(), pipe_gb.predict(X_test_full), average='micro')))
    # GradientBoost Classifier with Recursive Feature Elimination
    print('----------------------------With Recursive Feature Elimination--------------------------------------')
    pipe_gb_rfe = Pipeline([('scl', StandardScaler()), ('est', GradientBoostingClassifier(random_state=39))])
    pipe_gb_rfe.fit(X_train_rfe, y_train_full.values.ravel())
    print('Train Accuracy: {:.3f}'.format(accuracy_score(y_train_full.values.ravel(), pipe_gb_rfe.predict(X_train_rfe))))
    print('Test Accuracy: {:.3f}'.format(accuracy_score(y_test_full.values.ravel(), pipe_gb_rfe.predict(X_test_rfe))))
    print('Train F1 Score: {:.3f}'.format(f1_score(y_train_full.values.ravel(), pipe_gb_rfe.predict(X_train_rfe), average='micro')))
    print('Test F1 Score: {:.3f}'.format(f1_score(y_test_full.values.ravel(), pipe_gb_rfe.predict(X_test_rfe), average='micro')))

    cv = cross_val_score(pipe_gb, X_, y_.values.ravel(), cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=39))
    print('Cross validation with StratifiedKFold scores: {}'.format(cv))
    print('Cross Validation with StatifiedKFold mean: {}'.format(cv.mean()))

    # GridSearch
    n_features = len(df.columns)
    param_grid = {
        'learning_rate': [0.01, 0.1, 1, 10],
        'n_estimators': [1, 10, 100, 200, 300],
        'max_depth': [1, 2, 3, 4, 5]
    }
    stratifiedcv = StratifiedKFold(n_splits=10, shuffle=True, random_state=39)
    X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.33, random_state=42)

    grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=stratifiedcv)
    grid_search.fit(X_train, y_train.values.ravel())
    print('GridSearch Train Accuracy: {:.3f}'.format(accuracy_score(y_train.values.ravel(), grid_search.predict(X_train))))
    print('GridSearch Test Accuracy: {:.3f}'.format(accuracy_score(y_test.values.ravel(), grid_search.predict(X_test))))
    print('GridSearch Train F1 Score: {:.3f}'.format(f1_score(y_train.values.ravel(), grid_search.predict(X_train), average='micro')))
    print('GridSearch Test F1 Score: {:.3f}'.format(f1_score(y_test.values.ravel(), grid_search.predict(X_test), average='micro')))

    # GridSearch results
    print("Best params:\n{}".format(grid_search.best_params_))
    print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
    results = pd.DataFrame(grid_search.cv_results_)
    corr_params = results.drop(results.columns[[0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20]], axis=1)
    corr_params.head()

    # GridSearch in nested
    cv_gb = cross_val_score(grid_search, X_, y_.values.ravel(), cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=39))
    print('Grid Search with nested cross validation scores: {}'.format(cv_gb))
    print('Grid Search with nested cross validation mean: {}'.format(cv_gb.mean()))

if __name__ == '__main__':
    main()

