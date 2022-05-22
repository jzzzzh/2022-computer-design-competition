import random

from flask import Flask, make_response, request
from flask import jsonify
from flask_cors import CORS
from flask import send_file, send_from_directory
import os












import csv
import keras.models
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
import numpy as np
import math
import tensorflow
import time
import random
import random as rand
from keras import backend as K
commission = 0.0
bitcoin = []
gold = []
bitcoinpred = []
goldpred = []
mypred2 = []
# money[usd, gold, bitcoin]
# money = [1000, 0, 0]
money = [1000, 0, 0]
m = 0
n = 0
def difference(data_set,interval=1):
    diff=list()
    for i in range(interval,len(data_set)):
        value=data_set[i]-data_set[i-interval]
        diff.append(value)
    return pd.Series(diff)
# 对预测的数据进行逆差分转换
def invert_difference(history, yhat, interval=1):
    return yhat + history[-interval]
# 将数据转换为监督学习集，移位后产生的NaN值补0
def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df
# 将数据缩放到[-1,1]之间
def scale(train, test):
    # 创建一个缩放器，将数据集中的数据缩放到[-1,1]的取值范围中
    scaler = MinMaxScaler(feature_range=(-1, 1))
    # 使用数据来训练缩放器
    scaler = scaler.fit(train)
    # 使用缩放器来将训练集和测试集进行缩放
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled
# 将预测值进行逆缩放，使用之前训练好的缩放器，x为一维数组，y为实数
def invert_scale(scaler, X, y):
    # 将X,y转换为一个list列表
    new_row = [x for x in X] + [y]
    # 将列表转换为数组
    array = np.array(new_row)
    # 将数组重构成一个形状为[1,2]的二维数组->[[10,12]]
    array = array.reshape(1, len(array))
    # 逆缩放输入的形状为[1,2]，输出形状也是如此
    invert = scaler.inverse_transform(array)
    # 只需要返回y值即可
    return invert[0, -1]
# 构建一个LSTM模型
def fit_lstm(train, batch_size, nb_epoch, neurons):
    # 将数据对中的x和y分开
    X, y = train[:, 0:-1], train[:, -1]

    # 将2D数据拼接成3D数据，形状为[N*1*1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    print(batch_size, X.shape[1], X.shape[2])
    model = Sequential()
    # model = keras.models.Sequential()
    print(neurons)
    # model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dense(1))
    print(model.summary())
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        # shuffle是不混淆数据顺序
        his = model.fit(X, y, batch_size=batch_size, verbose=1, shuffle=False)
        # 每训练完一次就重置一次网络状态，网络状态与网络权重不同
        model.reset_states()
    return model
# 开始单步预测
def forecast_lstm(model, batch_size, X):
    # 将形状为[1:]的，包含一个元素的一维数组X，转换形状为[1,1,1]的3D张量
    X = X.reshape(1, 1, len(X))
    # 输出形状为1行一列的二维数组yhat
    yhat = model.predict(X, batch_size=batch_size)
    # 将yhat中的结果返回
    return yhat[0, 0]
    # return yhat
def get_bitcoin_pred(testNum,Number_of_samples,train_times,Number_of_neurons):
    data = pd.read_csv("bitcoin.csv")
    print(data.head())
    series = data.set_index(['date'], drop=True)
    raw_value = series.values
    print(raw_value)
    diff_value = difference(raw_value,1)
    # print("gggggg")
    print(diff_value)
    supervised = timeseries_to_supervised(diff_value,1)
    supervised_value = supervised.values
    # testNum = 1823
    # if(testNum > 5):
    #     testNum = testNum-5
    # print("hhhhh")
    print(supervised_value)
    # print("iiii")
    train, test = supervised_value[:-testNum], supervised_value[-testNum:]
    print("train")
    print(train)
    print("test")
    print(test)
    # 将训练集和测试集都缩放到[-1,1]之间
    scaler, train_scaled, test_scaled = scale(train, test)

    # print(train_scaled)
    X,y = train[:,0:-1],train[:,-1]
    X= X.reshape(X.shape[0],1,X.shape[1])
    # print(X)
    # # 构建一个LSTM模型并训练，样本数为1，训练次数为5，LSTM层神经元个数为4
    lstm_model = fit_lstm(train_scaled, Number_of_samples,train_times,Number_of_neurons)
    predictions = list()
    for i in range(len(test_scaled)):
        # 将测试集拆分为X和y
        # print(test)
        X, y = test[i, 0:-1], test[i, -1]
        # 将训练好的模型、测试数据传入预测函数中
        x=K.cast_to_floatx(X)
        # x=X
        # print(x)
        yhat = forecast_lstm(lstm_model, 1, x)
        # 将预测值进行逆缩放
        yhat = invert_scale(scaler, X, yhat)
        # 对预测的y值进行逆差分
        yhat = invert_difference(raw_value, yhat, len(test_scaled) + 1 - i)
        # 存储正在预测的y值
        predictions.append(yhat)
    rmse=mean_squared_error(raw_value[:testNum],predictions)
    print("Test RMSE:",rmse)
    print("bitcoin_pred")
    print(predictions)
    # with open('pred.csv','w',newline='')as f:
    #     f_csv = csv.writer(f)
    #     headers = ["bitcoin"]
    #     f_csv.writerow(headers)
    #     f_csv.writerows(predictions)

    plt.plot(raw_value[-testNum:])
    plt.plot(predictions)
    plt.legend(['true','pred'])
    plt.show()
    return predictions
def get_gold_pred(testNum,Number_of_samples,train_times,Number_of_neurons):
    # data = pd.read_csv("gold_refill.csv")
    data = pd.read_csv("gold.csv")
    print(data.head())
    series = data.set_index(['date'], drop=True)
    raw_value = series.values
    diff_value = difference(raw_value, 1)
    supervised = timeseries_to_supervised(diff_value, 1)
    supervised_value = supervised.values

    # testNum = 1823
    # if(testNum > 5):
    #     testNum = testNum-5
    # print("hhhhh")
    # print(supervised_value)
    train, test = supervised_value[:-testNum], supervised_value[-testNum:]

    # 将训练集和测试集都缩放到[-1,1]之间
    scaler, train_scaled, test_scaled = scale(train, test)
    # print(train_scaled)
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    # print(X)
    # # 构建一个LSTM模型并训练，样本数为1，训练次数为5，LSTM层神经元个数为4
    lstm_model = fit_lstm(train_scaled, Number_of_samples,train_times,Number_of_neurons)
    predictions = list()
    for i in range(len(test_scaled)):
        # 将测试集拆分为X和y
        # print(test)
        X, y = test[i, 0:-1], test[i, -1]
        # 将训练好的模型、测试数据传入预测函数中
        x = K.cast_to_floatx(X)
        # x=X
        # print(x)
        yhat = forecast_lstm(lstm_model, 1, x)
        # 将预测值进行逆缩放
        yhat = invert_scale(scaler, X, yhat)
        # 对预测的y值进行逆差分
        yhat = invert_difference(raw_value, yhat, len(test_scaled) + 1 - i)
        # 存储正在预测的y值
        predictions.append(yhat)
    rmse = mean_squared_error(raw_value[:testNum], predictions)
    print("Test RMSE:", rmse)
    print("gold_pred")
    print(predictions)
    # with open('predgold.csv', 'w', newline='') as f:
    #     f_csv = csv.writer(f)
    #     headers = ["gold"]
    #     f_csv.writerow(headers)
    #     f_csv.writerows(predictions)

    plt.style.use({'figure.figsize': (20, 8)})
    plt.plot(raw_value[-testNum:])
    plt.plot(predictions)
    plt.legend(['true', 'pred'])
    plt.show()
    return  predictions
def get_bitcoin_true():
    bitcoin_true = []
    with open('bitcoin.csv') as f:
        f_csv = csv.reader(f)
        list_f_csv = list(f_csv)
        for row in list_f_csv[1:]:
            bitcoin_true.append(row[1])
    return bitcoin_true
def get_gold_true():
    gold_true = []
    with open('gold.csv') as f:
        f_csv = csv.reader(f)
        list_f_csv = list(f_csv)
        for row in list_f_csv[1:]:
            gold_true.append(row[1])
    return gold_true
def get_bitcoin_differential_value(bitcoin_true):
    bitcoin_differential_value = []
    bitcoin_differential_value.append(0)
    for i in range(1,len(bitcoin_true)):
        bitcoin_differential_value.append(float(bitcoin_true[i])-float(bitcoin_true[i-1]))
    return bitcoin_differential_value
def get_gold_differential_value(gold_true):
    gold_differential_value = []
    gold_differential_value.append(0)
    for i in range(1,len(gold_true)):
        gold_differential_value.append(float(gold_true[i])-float(gold_true[i-1]))
    return gold_differential_value
def get_gold_growth(gold_true):
    gold_growth = []
    t_growth = 0
    gold_growth.append(t_growth)
    for i in range(1,len(gold_true)):
        if(float(gold_true[i]) >= float(gold_true[i-1])):
            t_growth = t_growth + 1
        else:
            t_growth = 0
        gold_growth.append(t_growth)
    return gold_growth
def get_bitcoin_growth(bitcoin_true):
    bitcoin_growth = []
    t_growth = 0
    bitcoin_growth.append(t_growth)
    for i in range(1, len(bitcoin_true)):
        if (float(bitcoin_true[i]) >= float(bitcoin_true[i - 1])):
            t_growth = t_growth + 1
        else:
            t_growth = 0
        bitcoin_growth.append(t_growth)
    return bitcoin_growth
def get_gold_descend(gold_true):
    gold_descend = []
    t_descend = 0
    gold_descend.append(t_descend)
    for i in range(1, len(gold_true)):
        if (float(gold_true[i]) <= float(gold_true[i - 1])):
            t_descend = t_descend + 1
        else:
            t_descend = 0
        gold_descend.append(t_descend)
    return gold_descend
def get_bitcoin_descend(bitcoin_true):
    bitcoin_descend = []
    t_descend = 0
    bitcoin_descend.append(t_descend)
    for i in range(1, len(bitcoin_true)):
        if (float(bitcoin_true[i]) <= float(bitcoin_true[i - 1])):
            t_descend = t_descend + 1
        else:
            t_descend = 0
        bitcoin_descend.append(t_descend)
    return bitcoin_descend

def write_mypred_csv(gold_true, bitcoin_true, gold_pred, bitcoin_pred,bitcoin_differential_value,gold_differential_value,gold_growth,bitcoin_growth,gold_descend,bitcoin_descend):
    with open('mypred2.csv', 'w', newline='') as f:
        f_csv = csv.writer(f)
        headers = ["", "num", "BitcoinValue", "GoldValue", "bitcoinPred", "goldPred", "mark",
                   "bitcoin differential value", "gold differential value", "gold growth", "bitcoin growth",
                   "gold descend", "bitcoin descend"]
        f_csv.writerow(headers)
        print(len(gold_true))
        print(len(bitcoin_true))
        print(len(gold_pred))
        print(len(bitcoin_pred))
        for i in range(0, len(gold_pred)):
            listtmp = [i,i,bitcoin_true[i],gold_true[i], bitcoin_pred[i][0], gold_pred[i][0], "TRUE",bitcoin_differential_value[i],gold_differential_value[i],gold_growth[i],bitcoin_growth[i],gold_descend[i],bitcoin_descend[i]]
            f_csv.writerow(listtmp)
def read():
    with open('mypred2.csv') as f:
        f_csv = csv.reader(f)
        list_f_csv = list(f_csv)
        for row in list_f_csv[1:]:
            x = []
            #bitcoin differential value	  gold differential value	  gold growth
            # bitcoin growth	 gold descend	bitcoin descend
            x.append(row[7])
            x.append(row[8])
            x.append(row[9])
            x.append(row[10])
            x.append(row[11])
            x.append(row[12])
            mypred2.append(x)
    with open('mypred2.csv') as f:
        f_csv = csv.reader(f)
        list_f_csv = list(f_csv)
        # print(list_f_csv)
        # date = list_f_csv[1][1]
        # print(date)
        # print(list_f_csv[1][4])
        for row in list_f_csv[1:]:
            x = []
            x.append(row[1])
            x.append(row[2])
            bitcoin.append(x)
            y = []
            y.append(row[1])
            y.append(row[3])
            y.append(row[6])
            gold.append(y)
            bitcoinpred.append(row[4])
            goldpred.append(row[5])
    # with open('pred.csv') as f:
    #     f_csv = csv.reader(f)
    #     list_f_csv = list(f_csv)
    #     # print(list_f_csv)
    #     date = list_f_csv[1][3]
    #     print(date)
    #     print(list_f_csv[1][4])
    #     for row in list_f_csv[1:]:
    #         bitcoinpred.append(row[3])
    #         goldpred.append(row[4])
    #     # print(bitcoinpred)
    #     # print(goldpred)
    # with open('BCHAIN-MKPRU.csv') as f:
    #     f_csv = csv.reader(f)
    #     list_f_csv = list(f_csv)
    #     date = list_f_csv[1][0]
    #     for row in list_f_csv[2:]:
    #         bitcoin.append(row)
    # with open('gold_refill.csv') as f:
    #     f_csv = csv.reader(f)
    #     list_f_csv = list(f_csv)
    #     date = list_f_csv[1][0]
    #     for row in list_f_csv[2:]:
    #         gold.append(row)
    # # print(bitcoin)
    # # print(gold)
def buy_bitcoin(x,n,bitcoinCommission):
    #money is the whole money, x is the money buy bitcoin, n is the day
    if float(money[0]) >= x:
        true_buy_money = (1-bitcoinCommission) * x
        bitcoin_num = true_buy_money/float(bitcoin[n][1])
        money[0] = money[0] - x
        money[2] = money[2] + bitcoin_num
    else:
        print("error,money buy bitcoin is not enough")
def sell_bitcoin(y,n,bitcoinCommission):
    #money is the whole money, y is the money sell bitcoin, n is the day
    if float(money[2]) >= y:
        true_sell_money = (1-bitcoinCommission) * y
        money_num = true_sell_money*float(bitcoin[n][1])
        money[0] = money[0] + money_num
        money[2] = money[2] - y
    else:
        print("error,money sell bitcoin is not enough")
def sell_gold(y, m,goldCommission):
    # money is the whole money, y is the money sell gold, m is the day
    if float(money[1]) >= y:
        true_sell_money = (1-goldCommission) * y
        money_num = true_sell_money * float(gold[n][1])
        money[0] = money[0]+ money_num
        money[1] = money[1] - y
    else:
        print("error,money sell gold is not enough")
def buy_gold(x,m,goldCommission):
    #money is the whole money, x is the money buy gold, m is the day
    if float(money[0]) >= x:
        true_buy_money = (1-goldCommission) * x
        gold_num = true_buy_money/float(gold[n][1])
        money[0] = money[0] - x
        money[1] = money[1] + gold_num
    else:
        print("error,money buy gold is not enough")
def showWholeMoney(m,n):
    #money is the whole money, m is the gold day, n is the bitcoin day,
    # print(money)
    whole_money = money[0] + money[1]*float(gold[n][1]) + money[2]*float(bitcoin[n][1])
    # whole_money = money[0] + money[2]*float(bitcoin[n+1][1])
    # whole_money = money[0] + money[1]*float(gold[m+1][1])
    # print(float(gold[m+1][1]))
    # print(m)
    # print(gold[m+1])
    # print('$  ' + str(whole_money))
    return whole_money


# 0手上美元，1手上黄金，2手上比特币，3现今价格，4预测价格，5预测与现价之差，6涨跌天数（正为涨，负为跌）
# 7最大值，8最小值，9平均值，10[40天值]，11[40天间隔一天的差值（39个）前减后]
# ，12[40天间隔两天差值（38个）前减后]，
# 13[40天间隔10天差值（30个）前减后],14 最大值时间和最小值时间差值
def getGoldlist(i,money,listPast,listFuture,recentMoney):
    # 0手上美元，1手上黄金，2手上比特币，3现今价格，4预测价格，5预测与现价之差，6涨跌天数（正为涨，负为跌）
    # 7最大值，8最小值，9平均值，10[40天值]，11[40天间隔一天的差值（39个）前减后]
    # ，12[40天间隔两天差值（38个）前减后]，
    # 13[40天间隔10天差值（30个）前减后],14 最大值时间和最小值时间差值
    list = []
    dollarnum = float(money[0])
    goldnum = float(money[1])
    BTCnum = float(money[2])
    recMon = float(recentMoney)
    preMon = float(listFuture)
    list.append(dollarnum)
    list.append(goldnum)
    list.append(BTCnum)
    list.append(recMon)
    list.append(preMon)

    list.append(float(mypred2[i][1]))
    list.append(float(mypred2[i][2]) - float(mypred2[i][4]))

    list1 = []
    list2 = []
    list3 = []
    list10 = []
    list20 = []
    list30 = []
    list4 = []
    list40 = []
    max = 0.0
    sum = 0.0
    min = 999999.99
    mnum = -1
    nnum = -1
    i = 0
    for a in listPast:
        i+=1
        e = float(a[1])
        sum += e
        if e > max:
            max = e
            mnum = i
        if e < min:
            min = e
            nnum = i
        list1.append(e)
    differtime = mnum - nnum
    #differtime > 0 总体上升 <0 总体下降
    list.append(max)
    list.append(min)
    list.append(sum / 40.0)
    for a in list1:
        list10.append(((float(a) / max) - 0.5) * 2)
    max = 0.0
    i = 0
    for a in listPast[1:]:
        i += 1
        b = listPast[i - 1]
        r = float(a[1]) - float(b[1])
        if r > max:
            max = r
        list2.append(r)
    # for a in list2:
    #     list20.append(((float(a) / max) - 0.5) * 2)
    i = 1
    max = 0.0
    for a in listPast[2:]:
        i += 1
        b = listPast[i - 2]
        t = float(a[1]) - float(b[1])
        if t > max:
            max = t
        list3.append(t)
    # for a in list3:
    #     list30.append(((float(a) / max) - 0.5) * 2)
    i = 9
    max = 0.0
    for a in listPast[10:]:
        i += 1
        b = listPast[i - 10]
        t = float(a[1]) - float(b[1])
        if t > max:
            max = t
        list4.append(t)
    list.append(list10)
    list.append(list2)
    list.append(list3)
    # list.append(list20)
    # list.append(list30)
    list.append(list4)
    # print(list4)
    # print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    list.append(differtime)
    list.append(i)
    return list

 # 0手上美元，1手上黄金，2手上比特币，3现今价格，4预测价格，5预测与现价之差，6涨跌天数（正为涨，负为跌）
    # 7最大值，8最小值，9平均值，10[40天值]，11[40天间隔一天的值（39个）]，12[40天间隔两天差值（38个）]
def getBTClist(i,money,listPast,listFuture,recentMoney):
    # 0手上美元，1手上黄金，2手上比特币，3现今价格，4预测价格，5预测与现价之差，6涨跌天数（正为涨，负为跌）
    # 7最大值，8最小值，9平均值，10[40天值]，11[40天间隔一天的值（39个）]，12[40天间隔两天差值（38个）]
    list = []
    dollarnum = float(money[0])
    goldnum = float(money[1])
    BTCnum = float(money[2])
    recMon = float(recentMoney)
    preMon = float(listFuture)
    list.append(dollarnum)
    list.append(goldnum)
    list.append(BTCnum)
    list.append(recMon)
    list.append(preMon)


    list.append(float(mypred2[i][0]))
    list.append(float(mypred2[i][3])-float(mypred2[i][5]))


    list1=[]
    list2=[]
    list3=[]
    list10 = []
    list20 = []
    list30 = []
    list4 = []
    list40 = []
    max = 0.0
    sum = 0.0
    min = 999999.99
    mnum = -1
    nnum = -1
    i = 0
    for a in listPast:
        i+=1
        e=float(a[1])
        sum+=e
        if e > max:
            max = e
            mnum = i
        if e < min:
            min = e
            nnum = i
        list1.append(e)
        differtime = mnum - nnum
    list.append(max)
    list.append(min)
    list.append(sum/40.0)
    # for a in list1:
    #     list10.append(((float(a)/max)-0.5)*2)
    max = 0.0
    i = 0
    for a in listPast[1:]:
        i+=1
        b = listPast[i-1]
        # print(a)
        # print(b)
        r=float(a[1])-float(b[1])
        if r > max:
            max = r
        list2.append(r)
    # for a in list2:
    #     list20.append(((float(a)/max)-0.5)*2)
    max = 0.0
    i = 1
    for a in listPast[2:]:
        i+=1
        b = listPast[i-2]
        t=float(a[1])-float(b[1])
        if t > max:
            max = t
        list3.append(t)
    i = 9
    max = 0.0
    for a in listPast[10:]:
        i += 1
        b = listPast[i - 10]
        t = float(a[1]) - float(b[1])
        if t > max:
            max = t
        list4.append(t)
    # for a in list3:
    #     list30.append(((float(a)/max)-0.5)*2)
    list.append(list10)
    list.append(list2)
    list.append(list3)
    # list.append(list20)
    # list.append(list30)
    list.append(list4)
    # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    # print(list4)
    # print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    list.append(differtime)
    list.append(i)
    return list

    # 0手上美元，1手上黄金，2手上比特币，3现今价格，
    # 4预测价格，5预测与现价之差，6涨跌天数（正为涨，负为跌）
    # 7最大值，8最小值，9平均值，
    # 10[40天值]，11[40天间隔一天的值（39个）]，
    # 12[40天间隔两天差值（38个）],
def getBTCback(list, Glist,numList):
    # buy  btc
    # 1. B现在值小于平均值 (list[3] < list[9])
    # 2. B预测明天会涨(list[5] > 0)
    # 3. B涨的天数大于3天，小于6天(list[6] > 3 and list[6] < 6)
    # 4. B跌的天数超过6天(list[6]<= -6)
    # 5. B最近10天差值在跌，最大时间与最小时间差值在上升(list[13][29] > 0 and list[14] > 0)???????????
    # 6. G现在值大于平均值（Glist[3] > list[9])
    # 7. G预测明天会跌(Glist[5] > 0)
    # 8. G跌的天数大于3天，小于6天(Glist[6] < -3 and Glist[6] > -6)
    # 9. G涨的天数超过6天(Glist[6]>= 6)
    #10. G最近10天差值在涨，最大时间与最小时间差值在下降(Glist[13][29] < 0 and list[14] < 0)??????????
    #11. B上升迅猛程度(list[5]/list[3] > 0.08)
    #12. G下降迅猛程度(Glist[5]/Glist[3] < -0.08)
    # sell  btc
    # 1. B现在值大于平均值 (list[3] > list[9])
    # 2. B预测明天会跌(list[5] < 0)
    # 3. B跌的天数大于3天，小于6天(list[6] < -3 and list[6] > -6)
    # 4. B涨的天数超过6天(list[6]>= 6)
    # 5. B最近10天差值在涨，最大时间与最小时间差值在下降(list[13][29] < 0 and list[14] < 0)???????????
    # 6. G现在值小于平均值（Glist[3] < list[9])
    # 7. G预测明天会涨(Glist[5] < 0)
    # 8. G涨的天数大于3天，小于6天(Glist[6] > 3 and Glist[6] < 6)
    # 9. G跌的天数超过6天(Glist[6]<= -6)
    # 10. G最近10天差值在跌，最大时间与最小时间差值在上升(Glist[13][29] > 0 and list[14] > 0)??????????
    # 11. B下降迅猛程度(list[5]/list[3] < -0.08)
    # 12. G上升迅猛程度(Glist[5]/Glist[3] > 0.08)

    # 0手上美元，1手上黄金，2手上比特币，3现今价格，4预测价格，5预测与现价之差，6涨跌天数（正为涨，负为跌）
    # 7最大值，8最小值，9平均值，10[40天值]，11[40天间隔一天的差值（39个）前减后]
    # ，12[40天间隔两天差值（38个）前减后]，
    # 13[40天间隔10天差值（30个）前减后],14 最大值时间和最小值时间差值,15时间
    num = 1
    #BTC
    # numList = [3,2,7,1,7,1,2,8,9,3,1,6]
    #判断的权重
    # numList = [2,	3.285714286,	7.571428571,	4.571428571	,7.714285714	,2.5	,4.5	,5.642857143,	6.714285714	,2.071428571,	4.785714286	,7.071428571]
    nusum = 0
    buywhat = 0.0
    sellwhat = 0.0
    flag = True
    for nu in numList:
        nusum=nu+nusum
    if(list[3] < list[9]):
        buywhat = buywhat + numList[0]/nusum
    elif(list[3] > list[9]):
        sellwhat = sellwhat + numList[0]/nusum
    if (list[5] > 0):
        buywhat = buywhat + numList[1] / nusum
    elif(list[5] < 0):
        sellwhat = sellwhat + numList[1] / nusum
    if (list[6] > 3 and list[6] < 6):
        buywhat = buywhat + numList[2] / nusum
    elif(list[6] < -3 and list[6] > -6):
        sellwhat = sellwhat + numList[2] / nusum
    if (list[6]<= -6):
        buywhat = buywhat + numList[3] / nusum
    elif(list[6]>= 6):
        sellwhat = sellwhat + numList[3] / nusum
    if (list[13][29] > 0 and list[14] > 0):
        buywhat = buywhat + numList[4] / nusum
    elif(list[13][29] < 0 and list[14] < 0):
        sellwhat = sellwhat + numList[4] / nusum
    if (Glist[3] > list[9]):
        buywhat = buywhat + numList[5] / nusum
    elif(Glist[3] < list[9]):
        sellwhat = sellwhat + numList[5] / nusum
    if (Glist[5] > 0):
        buywhat = buywhat + numList[6] / nusum
    elif(Glist[5] < 0):
        sellwhat = sellwhat + numList[6] / nusum
    if (Glist[6] < -3 and Glist[6] > -6):
        buywhat = buywhat + numList[7] / nusum
    elif(Glist[6] > 3 and Glist[6] < 6):
        sellwhat = sellwhat + numList[7] / nusum
    if (Glist[6]>= 6):
        buywhat = buywhat + numList[8] / nusum
    elif(Glist[6]<= -6):
        sellwhat = sellwhat + numList[8] / nusum
    if (Glist[13][29] < 0 and list[14] < 0):
        buywhat = buywhat + numList[9] / nusum
    elif(Glist[13][29] > 0 and list[14] > 0):
        sellwhat = sellwhat + numList[9] / nusum
    if (list[5]/list[3] > 0.08):
        buywhat = buywhat + numList[10] / nusum
    elif(list[5]/list[3] < -0.08):
        sellwhat = sellwhat + numList[10] / nusum
    if (Glist[5]/Glist[3] < -0.08):
        buywhat = buywhat + numList[11] / nusum
    elif(Glist[5]/Glist[3] > 0.08):
        sellwhat = sellwhat + numList[11] / nusum
    ratelist = [-0.8, -0.5, 0.8, 0.5, 0.01]
    #回撤比例和入场比例
    if(sellwhat > 0.5):
        num = ratelist[0]
        flag = True
    elif(sellwhat > 0.3):
        num = ratelist[1]
    elif buywhat > 0.5:
        num = ratelist[2]
    elif buywhat > 0.3:
        num = ratelist[3]
    else:
        num = ratelist[-1]
    if(flag == True):
        if(list[15] == 40):
            num = 0.7
            flag = False
        else:
            if(num > 0):
                num = 0.6
                flag = False
    # if(list[6] > 3):
    #     num = -0.2
    #     if (list[9] >= list[3]):
    #         num = num * 0.8
    #     else:
    #         num = num *1.1
    # elif(list[6] == 3):
    #     num = -0.1
    #     if (list[9] >= list[3]):
    #         num = num * 0.8
    #     else:
    #         num = num *1.1
    # elif(list[6] < -9):
    #     num = -0.8
    # elif(list[6] < -3):
    #     num = 0.9
    #     if (list[9] <= list[3]):
    #         num = num * 0.8
    #     else:
    #         num = num *1.1
    # elif(list[6] == -3):
    #     num = 0.8
    #     if (list[9] <= list[3]):
    #         num = num * 0.8
    #     else:
    #         num = num *1.1
    # if list[15] > 463 and list[15] < 563
    return num
def getGoldback(list, Glist,numList):
    # buy gold
    # 1. B现在值大于平均值 (list[3] > list[9])
    # 2. B预测明天会跌(list[5] < 0)
    # 3. B跌的天数大于3天，小于6天(list[6] < -3 and list[6] > -6)
    # 4. B涨的天数超过6天(list[6]>= 6)
    # 5. B最近10天差值在涨，最大时间与最小时间差值在下降(list[13][29] < 0 and list[14] < 0)???????????
    # 6. G现在值小于平均值（Glist[3] < list[9])
    # 7. G预测明天会涨(Glist[5] < 0)
    # 8. G涨的天数大于3天，小于6天(Glist[6] > 3 and Glist[6] < 6)
    # 9. G跌的天数超过6天(Glist[6]<= -6)
    # 10. G最近10天差值在跌，最大时间与最小时间差值在上升(Glist[13][29] > 0 and list[14] > 0)??????????
    # 11. B下降迅猛程度(list[5]/list[3] < -0.08)
    # 12. G上升迅猛程度(Glist[5]/Glist[3] > 0.08)
    # sell gold
    # 1. B现在值小于平均值 (list[3] < list[9])
    # 2. B预测明天会涨(list[5] > 0)
    # 3. B涨的天数大于3天，小于6天(list[6] > 3 and list[6] < 6)
    # 4. B跌的天数超过6天(list[6]<= -6)
    # 5. B最近10天差值在跌，最大时间与最小时间差值在上升(list[13][29] > 0 and list[14] > 0)???????????
    # 6. G现在值大于平均值（Glist[3] > list[9])
    # 7. G预测明天会跌(Glist[5] > 0)
    # 8. G跌的天数大于3天，小于6天(Glist[6] < -3 and Glist[6] > -6)
    # 9. G涨的天数超过6天(Glist[6]>= 6)
    # 10. G最近10天差值在涨，最大时间与最小时间差值在下降(Glist[13][29] < 0 and list[14] < 0)??????????
    # 11. B上升迅猛程度(list[5]/list[3] > 0.08)
    # 12. G下降迅猛程度(Glist[5]/Glist[3] < -0.08)

    # 0手上美元，1手上黄金，2手上比特币，3现今价格，4预测价格，5预测与现价之差，6涨跌天数（正为涨，负为跌）
    # 7最大值，8最小值，9平均值，10[40天值]，11[40天间隔一天的差值（39个）前减后]
    # ，12[40天间隔两天差值（38个）前减后]，
    # 13[40天间隔10天差值（30个）前减后],14 最大值时间和最小值时间差值
    num = 0.0
    # if (list[6] > 2):
    #     num = -0.08
    #     if(list[9] >= list[3]):
    #         num = num * 0.8
    #     else:
    #         num = num*1.1
    # elif (list[6] == 2):
    #     num = -0.05
    #     if (list[9] >= list[3]):
    #         num = num * 0.7
    #     else:
    #         num = num *1.1
    # elif (list[6] < -3):
    #     num = 0.15
    #     if (list[9] <= list[3]):
    #         num = num * 0.8
    #     else:
    #         num = num *1.1
    # elif (list[6] == -3):
    #     num = 0.4
    #     if (list[9] <= list[3]):
    #         num = num * 0.8
    #     else:
    #         num = num *1.1
    num = 1
    # gold
    #5.214285714	4.571428571	6.571428571	5.142857143	6.928571429	5.357142857	4	5.285714286	6	6	5.428571429	6.785714286
    # numList = [6, 5, 7, 1, 1, 2, 4, 8, 4, 4, 2, 8]
    flag = True
    # numList = [5.214285714	,4.571428571	,6.571428571,	5.142857143,	6.928571429	,5.357142857	,4,	5.285714286	,6,	6,	5.428571429,	6.785714286]
    nusum = 0
    buywhat = 0.0
    sellwhat = 0.0
    for nu in numList:
        nusum += nu
    if (list[3] < list[9]):
        buywhat = buywhat + numList[0] / nusum
    elif (list[3] > list[9]):
        sellwhat = sellwhat + numList[0] / nusum
    if (list[5] > 0):
        buywhat = buywhat + numList[1] / nusum
    elif (list[5] < 0):
        sellwhat = sellwhat + numList[1] / nusum
    if (list[6] > 3 and list[6] < 6):
        buywhat = buywhat + numList[2] / nusum
    elif (list[6] < -3 and list[6] > -6):
        sellwhat = sellwhat + numList[2] / nusum
    if (list[6] <= -6):
        buywhat = buywhat + numList[3] / nusum
    elif (list[6] >= 6):
        sellwhat = sellwhat + numList[3] / nusum
    if (list[13][29] > 0 and list[14] > 0):
        buywhat = buywhat + numList[4] / nusum
    elif (list[13][29] < 0 and list[14] < 0):
        sellwhat = sellwhat + numList[4] / nusum
    if (Glist[3] > list[9]):
        buywhat = buywhat + numList[5] / nusum
    elif (Glist[3] < list[9]):
        sellwhat = sellwhat + numList[5] / nusum
    if (Glist[5] > 0):
        buywhat = buywhat + numList[6] / nusum
    elif (Glist[5] < 0):
        sellwhat = sellwhat + numList[6] / nusum
    if (Glist[6] < -3 and Glist[6] > -6):
        buywhat = buywhat + numList[7] / nusum
    elif (Glist[6] > 3 and Glist[6] < 6):
        sellwhat = sellwhat + numList[7] / nusum
    if (Glist[6] >= 6):
        buywhat = buywhat + numList[8] / nusum
    elif (Glist[6] <= -6):
        sellwhat = sellwhat + numList[8] / nusum
    if (Glist[13][29] < 0 and list[14] < 0):
        buywhat = buywhat + numList[9] / nusum
    elif (Glist[13][29] > 0 and list[14] > 0):
        sellwhat = sellwhat + numList[9] / nusum
    if (list[5] / list[3] > 0.08):
        buywhat = buywhat + numList[10] / nusum
    elif (list[5] / list[3] < -0.08):
        sellwhat = sellwhat + numList[10] / nusum
    if (Glist[5] / Glist[3] < -0.08):
        buywhat = buywhat + numList[11] / nusum
    elif (Glist[5] / Glist[3] > 0.08):
        sellwhat = sellwhat + numList[11] / nusum
    ratelist = [-0.9, -0.5, 0.9, 0.5, 0.00]
    if (sellwhat > 0.5):
        num = ratelist[0]
    elif (sellwhat > 0.3):
        num = ratelist[1]
    elif buywhat > 0.5:
        num = ratelist[2]
    elif buywhat > 0.3:
        num = ratelist[3]
    else:
        num = ratelist[-1]
    if (flag == True):
        if (list[15] == 40):
            num = 0.4
            flag = False
        else:
            if (num > 0):
                num = 0.6
                flag = False
    return num
def buyHowManyB(i,money,listPast,listFuture,recentMoney,Glistpast,GlistFuture,GrecentMony,bitcoinCommission,numList):
    # print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    # print(money)
    # print(listPast)
    # print(listFuture)
    # print(recentMoney)
    # print(type(money))
    # print(type(listPast))
    # print(type(listFuture))
    # print(type(recentMoney))
    # print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    manylist = getBTClist(i, money, listPast, listFuture, recentMoney)
    Gmanylist = getGoldlist(i,money,Glistpast,GlistFuture,GrecentMony)
    many = getBTCback(manylist,Gmanylist,numList)
    # many = getBTCback1(manylist)
    x = random.randint(1,(25 - int(math.fabs(many))*10)*int(bitcoinCommission * 100))
    if math.fabs(many) < 0.4 and x > 1:
    # if math.fabs(many) < 0.6 and x > 10:
    # if x > 1:
        many = 0
    else:
        many = many
#many -1 ~ 1
    return many
def buyHowManyG(i,money,Blistpast,BlistFuture,BrecentMoney,listPast,listFuture,recentMoney,bitcoinCommission,numList):
    # print("ggggggggggggggggggggggggggggggggggggg")
    # print(money)
    # print(listPast)
    # print(listFuture)
    # print(recentMoney)
    # print("gggggggggggggggggggggggggggggggggggggg")

    manylist = getGoldlist(i,money,listPast,listFuture,recentMoney)
    Bmanylist = getBTClist(i,money,Blistpast,BlistFuture,BrecentMoney)
    many = getGoldback(Bmanylist,manylist,numList)
    x = random.randint(1, (25 - int(math.fabs(many))*10)*int(bitcoinCommission * 100))
    # many = getGoldback1(manylist)
    if math.fabs(many)< 0.4 and x > 1:
    # if math.fabs(many) < 0.6 and x > 1:
        many = 0
    else:
        many = many
    return many
def get_final_pred(test_time, bitcoinCommission,goldCommission,numListbit,numListgold):
    max = 0
    commission = 0.0
    # bitcoin = []
    # gold = []
    read()
    # # money[usd, gold, bitcoin]
    # money = [1000,0,0]
    #
    m = 0
    n = 0
    # bitcoinCommission = 0.02
    # goldCommission = 0.01
    # test_time = 1820

    USdollarlist = []
    for i in range(0, test_time):
        # what_to_do = rand.randint(1, 9)
        # what_to_do = 3
        # print("bitcoin = $" + str(bitcoin[n][1]))
        if (i >= 40):
            # print(i)
            # print("jjjjjjjjjjj")
            BTHlistpast = bitcoin[i - 40:i]
            BTHlistpred = bitcoinpred[i + 1]
            BTHdallor = bitcoin[n][1]
            GOLDlistpast = gold[i - 40:i]
            GOLDlistpred = goldpred[i + 1]
            GOLDdallor = gold[i][1]
            buyHowManyBTCnum = buyHowManyB(i, money, BTHlistpast, BTHlistpred, BTHdallor, GOLDlistpast, GOLDlistpred,GOLDdallor, bitcoinCommission,numListbit)
            buyHowManyGOLD = 0
            if gold[i][2] == 'TRUE':
                # print("gold = $" + str(gold[i][1]))
                buyHowManyGOLDnum = buyHowManyG(i, money, BTHlistpast, BTHlistpred, BTHdallor, GOLDlistpast,GOLDlistpred, GOLDdallor, goldCommission,numListgold)
            # else:
            #     print("sorry")
        else:
            buyHowManyGOLDnum = 0
            buyHowManyBTCnum = 0



        if buyHowManyGOLDnum == 0 and buyHowManyBTCnum == 0:
            # if buyHowManyGOLDnum >= 0:
            what_to_do = 1
        elif buyHowManyBTCnum == 0 and buyHowManyGOLDnum > 0:
            what_to_do = 2
        elif buyHowManyBTCnum > 0 and buyHowManyGOLDnum == 0:
            what_to_do = 3
        elif buyHowManyBTCnum > 0 and buyHowManyGOLDnum > 0:
            what_to_do = 4
        elif buyHowManyBTCnum == 0 and buyHowManyGOLDnum < 0:
            what_to_do = 5
        elif buyHowManyBTCnum < 0 and buyHowManyGOLDnum == 0:
            what_to_do = 6
        elif buyHowManyBTCnum < 0 and buyHowManyGOLDnum < 0:
            what_to_do = 7
        elif buyHowManyBTCnum < 0 and buyHowManyGOLDnum > 0:
            what_to_do = 8
        elif buyHowManyBTCnum > 0 and buyHowManyGOLDnum < 0:
            what_to_do = 9

        # what_to_do = 3
        # print(what_to_do)
        # if what_to_do == 1:  # 不动

            # print("No buy No sell")
        if what_to_do == 2:  # buy gold
            if gold[i][2] == 'TRUE':
                x = math.fabs(buyHowManyGOLDnum) * money[0]
                buy_gold(x, i, goldCommission)
                commission = commission + goldCommission * x
                # print("buy gold")
            # else:
            #     print("No buy No sell")
        elif what_to_do == 3:  # buy bitcoin
            x = math.fabs(buyHowManyBTCnum) * money[0]
            # print("jjjjjjjjjjj")
            # x = 0.5
            buy_bitcoin(x, n, bitcoinCommission)
            # print("jjjjjjjjjjj")
            commission = commission + bitcoinCommission * x
            # print("buy bitcoin")
        elif what_to_do == 4:  # buy both
            x = math.fabs(buyHowManyGOLDnum) * money[0]
            buy_bitcoin(x, n, bitcoinCommission)
            commission = commission + bitcoinCommission * x
            if gold[i][2] == 'TRUE':
                x = math.fabs(buyHowManyBTCnum) * money[0]
                buy_gold(x, i, goldCommission)
                commission = commission + goldCommission * x
                # print("buy both")
            # else:
            #     print("buy bitcoin")
        elif what_to_do == 5:  # sell gold
            if gold[i][2] == 'TRUE':
                y = math.fabs(buyHowManyGOLDnum) * money[1]
                commission = commission + goldCommission * y
                sell_gold(y, i, goldCommission)
                # print("sell gold")
            # else:
            #     print("No buy No sell")
        elif what_to_do == 6:  # sell bitcoin
            y = math.fabs(buyHowManyBTCnum) * money[2]
            commission = commission + bitcoinCommission * y
            sell_bitcoin(y, n, bitcoinCommission)
            # print("sell bitcoin")
        elif what_to_do == 7:  # sell both
            y = math.fabs(buyHowManyBTCnum) * money[2]
            commission = commission + bitcoinCommission * y
            sell_bitcoin(y, n, bitcoinCommission)
            if gold[i][2] == 'TRUE':
                y = math.fabs(buyHowManyGOLDnum) * money[1]
                commission = commission + goldCommission * y
                sell_gold(y, i, goldCommission)
                # print("sell both")
            # else:
            #     print("sell bitcoin")
        elif what_to_do == 8:  # buy GOLD and sell BTC
            y = math.fabs(buyHowManyBTCnum) * money[2]
            sell_bitcoin(y, n, bitcoinCommission)
            commission = commission + bitcoinCommission * y
            if gold[i][2] == 'TRUE':
                x = math.fabs(buyHowManyGOLDnum) * money[0]
                commission = commission + goldCommission * x
                buy_gold(x, i, goldCommission)
                # print("buy gold and sell BTC")
            # else:
            #     print("sell BTC")
        elif what_to_do == 9:  # buy BTC and sell GOLD
            if gold[i][2] == 'TRUE':
                x = math.fabs(buyHowManyGOLDnum) * money[1]
                commission = commission + goldCommission * x
                sell_gold(x, i, goldCommission)
                # print("buy BTC and sell GOLD")
            # else:
            #     print("buy BTC")
            y = math.fabs(buyHowManyBTCnum) * money[0]
            commission = commission + bitcoinCommission * y
            buy_bitcoin(y, n, bitcoinCommission)
        if gold[i][2] == 'TRUE':
            # if i % 7 == 1 or i % 7 == 2 or i % 7 == 3 or i%7 == 4 or i % 7 == 0:
            m = m + 1
        n = n + 1
        # print(str(n) + 'days')
        # print(i)
        USdollar = showWholeMoney(m, n)
        USdollarlist.append(USdollar)
        if max < USdollar:
            max = USdollar
    # plt.plot(USdollarlist, color="green")
    #
    # plt.show()
    # print("max= " + str(max))
    # print('commission=' + str(commission))
    # print(len(gold))
    # print(n)

    return USdollar
def get_final_EC(testNum,bitcoinCommission,goldCommission,ECtestNum,ECtesttimes,electRate,MutationRate, firstmoney):
    money = [firstmoney,0,0]
    bbestnumListgold = []
    bbestnumListbit = []
    maxUSD = -1;
    listgold = []
    listbit = []
    usdmoney = []
    bestnumListgold = []
    bestnumListbit = []
    electNum = int(ECtestNum*electRate)
    for tt in range(0, ECtestNum):
        numListgold = []
        numListbit = []
        for i in range(0, 12):
            numListgold.append(rand.randint(1, 10))
            numListbit.append(rand.randint(1, 10))
        listbit.append(numListbit)
        listgold.append(numListgold)
        usdmoney.append(get_final_pred(testNum, bitcoinCommission, goldCommission, numListbit, numListgold))
        money = [firstmoney, 0, 0]
    for aa in range(0, ECtestNum):
        max = -1;
        step = -1;
        for bb in range(aa, ECtestNum):
            if(max < usdmoney[bb]):
                step = bb
                max = usdmoney[bb]
        tmp = usdmoney[aa]
        usdmoney[aa] = usdmoney[step]
        usdmoney[step] = tmp

        tmp = listbit[aa]
        listbit[aa] = listbit[step]
        listbit[step] = tmp

        tmp = listgold[aa]
        listgold[aa] = listgold[step]
        listgold[step] = tmp
    bestnumListgold = listgold[0:electNum]
    bestnumListbit = listbit[0:electNum]
    for bb in range(0,ECtestNum-electNum):
        listg = []
        listb = []
        t1 = rand.randint(0,electNum)
        t2 = rand.randint(0,electNum)
        for cc in range(0,12):
            # print(cc)
            listb.append(int((listbit[t1][cc] + listbit[t2][cc]) / 2))
            listg.append(int((listgold[t1][cc] + listgold[t2][cc]) / 2))
        t3 = rand.randint(0,int(1/MutationRate)-1)
        if(t3 == 0):
            t4 = rand.randint(0,11)
            listb[t4] = listb[t4]+1
        bestnumListgold.append(listg)
        bestnumListbit.append(listb)
    for ttt in range(0,ECtesttimes-1):
        money = [firstmoney, 0, 0]
        usdmoney = []
        for tt in range(0, ECtestNum):
            usdmoney.append(get_final_pred(testNum, bitcoinCommission, goldCommission, bestnumListgold[tt], bestnumListbit[tt]))
            money = [firstmoney, 0, 0]
        for aa in range(0, ECtestNum):
            max = -1;
            step = -1;
            for bb in range(aa, ECtestNum):
                if (max < usdmoney[bb]):
                    step = bb
                    max = usdmoney[bb]
            tmp = usdmoney[aa]
            usdmoney[aa] = usdmoney[step]
            usdmoney[step] = tmp

            tmp = bestnumListbit[aa]
            bestnumListbit[aa] = bestnumListbit[step]
            bestnumListbit[step] = tmp

            tmp = bestnumListgold[aa]
            bestnumListgold[aa] = bestnumListgold[step]
            bestnumListgold[step] = tmp
        bestnumListgold = bestnumListgold[0:electNum]
        bestnumListbit = bestnumListbit[0:electNum]
        for bb in range(0, ECtestNum - electNum):
            listg = []
            listb = []
            t1 = rand.randint(0, electNum-1)
            t2 = rand.randint(0, electNum-1)
            for cc in range(0, 12):
                listb.append(int((bestnumListbit[t1][cc] + bestnumListbit[t2][cc]) / 2))
                listg.append(int((bestnumListgold[t1][cc] + bestnumListgold[t2][cc]) / 2))
            t3 = rand.randint(0, int(1 / MutationRate) - 1)
            if (t3 == 0):
                t4 = rand.randint(0, 11)
                listb[t4] = listb[t4] + 1
            bestnumListgold.append(listg)
            bestnumListbit.append(listb)


    bbestnumListbit = bestnumListbit[0]
    bbestnumListgold = bestnumListgold[0]



    with open('weight.csv', 'w', newline='') as f:
        f_csv = csv.writer(f)
        headers = ["bitcoinrate", "goldrate"]
        f_csv.writerow(headers)
        for i in range(0, 12):
            listtmp = [bbestnumListbit[i],bbestnumListgold[i]]
            f_csv.writerow(listtmp)
    print("finish")
def write_my_pred(testNum,Number_of_samples,train_times,Number_of_neurons):
    gold_true = get_gold_true()
    bitcoin_true = get_bitcoin_true()
    gold_pred = get_gold_pred(testNum, Number_of_samples, train_times, Number_of_neurons)
    bitcoin_pred = get_bitcoin_pred(testNum, Number_of_samples, train_times, Number_of_neurons)
    bitcoin_differential_value = get_bitcoin_differential_value(bitcoin_true)
    gold_differential_value = get_gold_differential_value(gold_true)
    gold_growth = get_gold_growth(gold_true)
    bitcoin_growth = get_bitcoin_growth(bitcoin_true)
    gold_descend = get_gold_descend(gold_true)
    bitcoin_descend = get_bitcoin_descend(bitcoin_true)

    write_mypred_csv(gold_true, bitcoin_true, gold_pred, bitcoin_pred, bitcoin_differential_value,
                     gold_differential_value, gold_growth, bitcoin_growth, gold_descend, bitcoin_descend)


def get_ffinal_pred(mymoney, testNum, bitcoinCommission, goldCommission, numListbit, numListgold):
    print("投资建议")
    max = 0
    commission = 0.0
    read()
    # # money[usd, gold, bitcoin]
    # money = [1000,0,0]
    #
    m = 0
    n = 0
    # bitcoinCommission = 0.02
    # goldCommission = 0.01
    # test_time = 1820
    finallyans = [0,0,0]
    USdollarlist = []
    i = testNum
    # what_to_do = rand.randint(1, 9)
    # what_to_do = 3
    print("bitcoin = $" + str(bitcoin[n][1]))
    if (i >= 0):
        # print(i)
        # print("jjjjjjjjjjj")
        BTHlistpast = bitcoin[i - 40:i]
        BTHlistpred = bitcoinpred[i + 1]
        BTHdallor = bitcoin[n][1]
        GOLDlistpast = gold[i - 40:i]
        GOLDlistpred = goldpred[i + 1]
        GOLDdallor = gold[i][1]
        buyHowManyBTCnum = buyHowManyB(i, mymoney, BTHlistpast, BTHlistpred, BTHdallor, GOLDlistpast, GOLDlistpred,
                                       GOLDdallor, bitcoinCommission, numListbit)
        buyHowManyGOLD = 0
        if gold[i][2] == 'TRUE':
            print("gold = $" + str(gold[i][1]))
            buyHowManyGOLDnum = buyHowManyG(i, mymoney, BTHlistpast, BTHlistpred, BTHdallor, GOLDlistpast,
                                            GOLDlistpred, GOLDdallor, goldCommission, numListgold)
        # else:
        #     print("sorry")
    else:
        buyHowManyGOLDnum = 0
        buyHowManyBTCnum = 0




    finallyans[1] = buyHowManyGOLDnum
    finallyans[2] = buyHowManyBTCnum
    if buyHowManyGOLDnum == 0 and buyHowManyBTCnum == 0:
        # if buyHowManyGOLDnum >= 0:
        what_to_do = 1
    elif buyHowManyBTCnum == 0 and buyHowManyGOLDnum > 0:
        what_to_do = 2
    elif buyHowManyBTCnum > 0 and buyHowManyGOLDnum == 0:
        what_to_do = 3
    elif buyHowManyBTCnum > 0 and buyHowManyGOLDnum > 0:
        what_to_do = 4
    elif buyHowManyBTCnum == 0 and buyHowManyGOLDnum < 0:
        what_to_do = 5
    elif buyHowManyBTCnum < 0 and buyHowManyGOLDnum == 0:
        what_to_do = 6
    elif buyHowManyBTCnum < 0 and buyHowManyGOLDnum < 0:
        what_to_do = 7
    elif buyHowManyBTCnum < 0 and buyHowManyGOLDnum > 0:
        what_to_do = 8
    elif buyHowManyBTCnum > 0 and buyHowManyGOLDnum < 0:
        what_to_do = 9
    finallyans[0] = what_to_do
    # what_to_do = 3
    # print(what_to_do)
    if what_to_do == 1:  # 不动
        print("No buy No sell")
    elif what_to_do == 2:  # buy gold
        if gold[i][2] == 'TRUE':
            x = math.fabs(buyHowManyGOLDnum) * money[0]
            print("x="+ str(x))
            print("buy gold")
        else:
            print("No buy No sell")
    elif what_to_do == 3:  # buy bitcoin
        x = math.fabs(buyHowManyBTCnum) * money[0]
        print("x=" + str(x))
        # print("jjjjjjjjjjj")
        # x = 0.5
        # buy_bitcoin(x, n, bitcoinCommission)
        # print("jjjjjjjjjjj")
        # commission = commission + bitcoinCommission * x
        print("buy bitcoin")
    elif what_to_do == 4:  # buy both
        x = math.fabs(buyHowManyGOLDnum) * money[0]
        print("x=" + str(x))
        # buy_bitcoin(x, n, bitcoinCommission)
        # commission = commission + bitcoinCommission * x
        if gold[i][2] == 'TRUE':
            x = math.fabs(buyHowManyBTCnum) * money[0]
            print("x=" + str(x))
            # buy_gold(x, i, goldCommission)
            # commission = commission + goldCommission * x
            print("buy both")
        else:
            print("buy bitcoin")
    elif what_to_do == 5:  # sell gold
        if gold[i][2] == 'TRUE':
            y = math.fabs(buyHowManyGOLDnum) * money[1]
            print("y=" + str(y))
            # commission = commission + goldCommission * y
            # sell_gold(y, i, goldCommission)
            print("sell gold")
        else:
            print("No buy No sell")
    elif what_to_do == 6:  # sell bitcoin
        y = math.fabs(buyHowManyBTCnum) * money[2]
        print("y=" + str(y))
        # commission = commission + bitcoinCommission * y
        # sell_bitcoin(y, n, bitcoinCommission)
        print("sell bitcoin")
    elif what_to_do == 7:  # sell both
        y = math.fabs(buyHowManyBTCnum) * money[2]
        print("y=" + str(y))
        # commission = commission + bitcoinCommission * y
        # sell_bitcoin(y, n, bitcoinCommission)
        if gold[i][2] == 'TRUE':
            y = math.fabs(buyHowManyGOLDnum) * money[1]
            print("y=" + str(y))
            # commission = commission + goldCommission * y
            # sell_gold(y, i, goldCommission)
            print("sell both")
        else:
            print("sell bitcoin")
    elif what_to_do == 8:  # buy GOLD and sell BTC
        y = math.fabs(buyHowManyBTCnum) * money[2]
        print("y=" + str(y))
        # sell_bitcoin(y, n, bitcoinCommission)
        # commission = commission + bitcoinCommission * y
        if gold[i][2] == 'TRUE':
            x = math.fabs(buyHowManyGOLDnum) * money[0]
            print("x="+str(x))
            # commission = commission + goldCommission * x
            # buy_gold(x, i, goldCommission)
            print("buy gold and sell BTC")
        else:
            print("sell BTC")
    elif what_to_do == 9:  # buy BTC and sell GOLD
        if gold[i][2] == 'TRUE':
            x = math.fabs(buyHowManyGOLDnum) * money[1]
            print("x = "+str(x))
            # commission = commission + goldCommission * x
            # sell_gold(x, i, goldCommission)
            print("buy BTC and sell GOLD")
        else:
            print("buy BTC")
        y = math.fabs(buyHowManyBTCnum) * money[0]
        print("y=" + str(y))
        # commission = commission + bitcoinCommission * y
        # buy_bitcoin(y, n, bitcoinCommission)
    return finallyans


def start_pred(mymoney,testNum,bitcoinCommission,goldCommission):
    # readddd!!!
    numListbit = []
    numListgold = []
    with open('weight.csv') as f:
        f_csv = csv.reader(f)
        list_f_csv = list(f_csv)
        for row in list_f_csv[1:]:
            numListbit.append(int(row[0]))
            numListgold.append(int(row[1]))
    aans = get_ffinal_pred(mymoney, testNum, bitcoinCommission, goldCommission, numListbit, numListgold)
    return aans

def start_EC(testNum,bitcoinCommission,goldCommission,ECtestNum,ECtesttimes,electRate, MutationRate, firstmoney):
    get_final_EC(testNum,bitcoinCommission,goldCommission,ECtestNum,ECtesttimes,electRate, MutationRate, firstmoney)


# if __name__ == "__main__":
#     while(1):
#         print("请选择需要的功能")
#         print("请确保在该程序文件夹下有gold.csv和bitcoin.csv")
#         print("gold.csv和bitcoin.csv数据是否更新")
#         isUpdate = input("Y/N")
#         if(isUpdate=="Y"):
#             testNum = input("有多少条数据")
#             Number_of_samples = 1
#             train_times = input("您想要的训练次数")
#             Number_of_neurons = input("LSTM层神经元个数")
#             print("更新中...")
#             write_my_pred(int(testNum), Number_of_samples, int(train_times), int(Number_of_neurons))
#
#
#         elif(isUpdate == "N"):
#             print("1. 进化计算算出最佳条件")
#             print("2. 进行今日决策")
#             user_choice = input(":")
#             if(user_choice == "1"):
#                 print("请确保在该程序文件夹下有gold.csv和bitcoin.csv")
#                 user_ans = input("Y/N")
#                 if(user_ans == "Y" or user_ans == "y"):
#                     ECtestNum = input("族群数:")
#                     ECtesttimes = input("遗传次数:")
#                     electRate = input("精英主义概率:")
#                     MutationRate = input("变异几率:")
#                     testNum = input("有多少条数据:")
#                     firstmoney = input("请输入本金数:")
#                     money = [float(firstmoney), 0, 0]
#                     bitcoinCommission = input("比特币的手续费（即本金乘以多少，例如0.01，请不要输入0）:")
#                     goldCommission = input("黄金的手续费（即本金乘以多少，例如0.01，请不要输入0）:")
#                     start_EC(int(testNum), float(bitcoinCommission),float(goldCommission),int(ECtestNum),int(ECtesttimes),float(electRate), float(MutationRate),float(firstmoney))
#                 else:
#                     print("please retry")
#             elif(user_choice=="2"):
#                 print("请确保在该程序文件夹下有weight.csv，gold.csv和bitcoin.csv")
#                 print("weight.csv若没有，可以通过1自动生成，但您需要提供gold.csv和bitcoin.csv")
#                 user_ans = input("Y/N")
#                 if (user_ans == "Y" or user_ans == "y"):
#                     firstmoney = input("请输入本金数:")
#                     secondgold = input("请输入黄金数:")
#                     thirdbitcoin = input("请输入比特币数:")
#                     mymoney = [float(firstmoney), float(secondgold), float(thirdbitcoin)]
#                     testNum = input("有多少条数据:")
#                     bitcoinCommission = input("比特币的手续费（即本金乘以多少，例如0.01，请不要输入0）:")
#                     goldCommission = input("黄金的手续费（即本金乘以多少，例如0.01，请不要输入0）:")
#                     start_pred(mymoney,int(testNum),float(bitcoinCommission),float(goldCommission))
#                 else:
#                     print("please retry")
#             else:
#                 print("请重新输入")












































































app = Flask(__name__)
CORS(app, resources=r'/*')

def whatTodo():
    return random.randint(0,9)


@app.route('/predict', methods=['GET', 'POST'])
def run():
    getJson = request.get_json()
    item1=str(getJson.get('item1'))
    item2=str(getJson.get('item2'))
    USD=str(getJson.get('USD'))
    item1money=str(getJson.get("item1money"))
    item2money=str(getJson.get("item2money"))
    MoneyForm=str(getJson.get("MoneyForm"))
    choose=str(getJson.get("choose"))
    write_my_pred(1000, 1, 4, 3)
    start_EC(500, 0.01, 0.01, 100, 2,
             0.1, 0.1, 1000)
    mymoney = [float(USD), float(item1money), float(item2money)]
    testNum = 500
    test = start_pred(mymoney, int(testNum), 0.01, 0.01)
    whatTodo = test[0]
    item1data = test[1]
    item2data = test[2]
    # whatTodo = random.randint(1, 9)
    # item1data = 0.1
    # item2data = 0.1
    if USD != "":
        res = {
            'code': 0,
            'msg': "OK",
            'data': {
                'whatToDo': whatTodo,
                'buyItem1': item1data,
                'buyItem2': item2data
            }
        }
    else:
        res = {
            'code': 1,
            'msg': 'err',
        }
    return make_response(res)



@app.route("/download", methods=['GET'])
def download_file():
    # 需要知道2个参数, 第1个参数是本地目录的path, 第2个参数是文件名(带扩展名)
    directory = os.getcwd()  # 假设在当前目录
    filename = "LSTM_GA_Predict.exe"
    return send_from_directory(directory, filename, as_attachment=True)

if __name__ == '__main__':
    app.run()
