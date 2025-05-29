from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf #ACF与PACF
from statsmodels.tsa.arima.model import ARIMA #ARIMA模型
from statsmodels.graphics.api import qqplot  #qq图
from scipy import stats

import warnings
warnings.filterwarnings("ignore")


# 绘图设置（适用于mac）
# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 模型检测
def Model_checking(model) -> None:
    # 残差检验:检验残差是否服从正态分布，画图查看，然后检验

    print('------------残差检验-----------')
    # model.resid：残差 = 实际观测值 – 模型预测值
    print(stats.normaltest(model.resid))

    # QQ图看正态性
    qqplot(model.resid, line="q", fit=True)
    plt.title("Q-Q图")
    plt.show()
    # 绘制直方图
    plt.hist(model.resid, bins=50)
    plt.show()

    # 进行Jarque-Bera检验:判断数据是否符合总体正态分布
    jb_test = sm.stats.stattools.jarque_bera(model.resid)
    print("==================================================")
    print('------------Jarque-Bera检验-----------')
    print('Jarque-Bera test:')
    print('JB:', jb_test[0])
    print('p-value:', jb_test[1])
    print('Skew:', jb_test[2])
    print('Kurtosis:', jb_test[3])

    # 残差序列自相关：残差序列是否独立
    print("==================================================")
    print('------DW检验:残差序列自相关----')
    print(sm.stats.stattools.durbin_watson(model.resid.values))

# 使用BIC矩阵计算p和q的值
def cal_pqValue(D_data, diff_num=0) -> list[float]:
    # 定阶
    pmax = int(len(D_data) / 10)  # 一般阶数不超过length/10
    qmax = int(len(D_data) / 10)  # 一般阶数不超过length/10
    bic_matrix = []  # BIC矩阵
    # 差分阶数
    diff_num = 2

    for p in range(pmax + 1):
        tmp = []
        for q in range(qmax + 1):
            try:
                tmp.append(ARIMA(D_data, order=(p, diff_num, q)).fit().bic)
            except Exception as e:
                print(e)
                tmp.append(None)
        bic_matrix.append(tmp)

    bic_matrix = pd.DataFrame(bic_matrix)  # 从中可以找出最小值
    p, q = bic_matrix.stack().idxmin()  # 先用stack展平，然后用idxmin找出最小值位置。
    print(u'BIC最小的p值和q值为:%s、%s' % (p, q))
    return p, q

# 计算时序序列模型
def cal_time_series(data, forecast_num=3) -> None:
    # 绘制时序图
    data.plot()
    # 存储图片
    plt.savefig('D:\\SmartAlarm\\相关材料\\ARIMA模型\\ARIMA\\ARIMA\\download')
    plt.show()

    # 绘制自相关图
    plot_acf(data).show()
    # 绘制偏自相关图
    plot_pacf(data).show()

    # 时序数据平稳性检测
    original_ADF = ADF(data[u'deal_data'])
    print(u'原始序列的ADF检验结果为：', original_ADF)

    # 对数序数据进行d阶差分运算，化为平稳时间序列
    diff_num = 0 # 差分阶数
    diff_data = data     # 差分数序数据
    ADF_p_value = ADF(data[u'deal_data'])[1]
    while  ADF_p_value > 0.01:
        diff_data = diff_data.diff(periods=1).dropna()
        diff_num = diff_num + 1
        ADF_result = ADF(diff_data[u'deal_data'])
        ADF_p_value = ADF_result[1]
        print("ADF_p_value:{ADF_p_value}".format(ADF_p_value=ADF_p_value))
        print(u'{diff_num}差分的ADF检验结果为：'.format(diff_num = diff_num), ADF_result )

    # 白噪声检测
    print(u'差分序列的白噪声检验结果为：', acorr_ljungbox(diff_data, lags=1))  # 返回统计量和p值

    # 使用AIC和BIC准则定阶q和p的值(推荐)
    AIC = sm.tsa.stattools.arma_order_select_ic(diff_data, max_ar=4, max_ma=4, ic='aic')['aic_min_order']
    BIC = sm.tsa.stattools.arma_order_select_ic(diff_data, max_ar=4, max_ma=4, ic='bic')['bic_min_order']
    print('---AIC与BIC准则定阶---')
    print('the AIC is{}\nthe BIC is{}\n'.format(AIC, BIC), end='')
    p = BIC[0]
    q = BIC[1]

    # 使用BIC矩阵来计算q和p的值
    # pq_result = cal_pqValue(diff_data, diff_num)
    # p = pq_result[0]
    # q = pq_result[1]

    # 构建时间序列模型
    model = ARIMA(data, order=(p, diff_num, q)).fit()  # 建立ARIMA(p, diff+num, q)模型
    print('模型报告为：\n', model.summary())
    print("预测结果：\n", model.forecast(forecast_num))

    print("预测结果(详细版)：\n")
    forecast = model.get_forecast(steps=forecast_num)
    table = pd.DataFrame(forecast.summary_frame())
    print(table)

    # 绘制残差图
    diff_data.plot(color='orange', title='残差图')
    model.resid.plot(figsize=(10, 3))
    plt.title("残差图")
    # plt.savefig('/Users/mac/Downloads/1.png')
    plt.show()

    # 模型检查
    Model_checking(model)


if __name__ == '__main__':
    # # 数据测试1:
    # need_data = {'2016-02': 44964.03, '2016-03': 56825.51, '2016-04': 49161.98, '2016-05': 45859.35,
    #              '2016-06': 45098.56,
    #              '2016-07': 45522.17, '2016-08': 57133.18, '2016-09': 49037.29, '2016-10': 43157.36,
    #              '2016-11': 48333.17,
    #              '2016-12': 22900.94,
    #              '2017-01': 67057.29, '2017-02': 49985.29, '2017-03': 49771.47, '2017-04': 35757.0, '2017-05': 42914.27,
    #              '2017-06': 44507.03, '2017-07': 40596.51, '2017-08': 52111.75, '2017-09': 49711.18,
    #              '2017-10': 45766.09,
    #              '2017-11': 45273.82, '2017-12': 22469.57,
    #              '2018-01': 71032.23, '2018-02': 37874.38, '2018-03': 44312.24, '2018-04': 39742.02,
    #              '2018-05': 43118.38,
    #              '2018-06': 33859.69, '2018-07': 38910.89, '2018-08': 39138.42, '2018-09': 37175.03,
    #              '2018-10': 44159.96,
    #              '2018-11': 46321.72, '2018-12': 22410.88,
    #              '2019-01': 61241.94, '2019-02': 31698.6, '2019-03': 44170.62, '2019-04': 47627.13, '2019-05': 54407.37,
    #              '2019-06': 50231.68, '2019-07': 61010.29, '2019-08': 59782.19, '2019-09': 57245.15,
    #              '2019-10': 61162.55,
    #              '2019-11': 52398.25, '2019-12': 15482.64,
    #              '2020-01': 38383.97, '2020-02': 26943.55, '2020-03': 57200.32, '2020-04': 49449.95,
    #              '2020-05': 47009.84,
    #              '2020-06': 49946.25, '2020-07': 56383.23, '2020-08': 60651.07}
    # data = {'time_data': list(need_data.keys()), 'deal_data': list(need_data.values())}
    # df = pd.DataFrame(data)
    # df.set_index(['time_data'], inplace=True)  # 设置索引
    # cal_time_series(df, 7) # 模型调用
    #数据测试2（从excel中读取）:
    path = 'D:\\SmartAlarm\\相关材料\\ARIMA模型\\ARIMA\\ARIMA\\时间序列模型测试数据.xlsx'
    df = pd.read_excel(path)
    df.rename(columns={'data': 'deal_data', 'time': 'time_data'}, inplace=True)
    df.set_index(['time_data'], inplace=True)  # 设置索引

    cal_time_series(df, 7) # 模型调用