"""
@version: 0.1
@author: Blade He
@license: Morningstar 
@contact: blade.he@morningstar.com
@site: 
@software: PyCharm
@file: main.py
@time: 2018/11/26
"""
from util.logutil import logger
import pandas as pd
import numpy as np
import math
import os
import time
import traceback


def startjob():
    outputfolder = './output/'
    rawfilefolder = './rawfile/'
    rawdatafile = 'effectivedatestudy.xlsx'
    # rawdatafile = 'minisample.xlsx'
    rawdatapath = os.path.join(rawfilefolder, rawdatafile)
    # outputfile = 'staticticsresult_mini.xlsx'
    outputfile = 'staticticsresult.xlsx'
    outputpath = os.path.join(outputfolder, outputfile)
    logger.info('Read data begin')
    # SecId Date    SRRI    Source
    rawdata = pd.read_excel(rawdatapath,
                            encoding='utf-8',
                            sheet_name='Sheet1')
    logger.info('Read data end')

    logger.info('Insert source data for KIID & EMT begin')
    dfbothkiidempt = rawdata[(rawdata['Source'] == 'KIID & EMT')]
    logger.info('There are {0} records with KIID & EMT'.format(len(dfbothkiidempt)))
    amount = len(rawdata)
    count = 1
    for index, row in dfbothkiidempt.iterrows():
        logger.info('Insert the {0} record for KIID & EMT'.format(count))
        rawdata.loc[amount] = {'SecId': dfbothkiidempt.loc[index, 'SecId'],
                                         'Date': dfbothkiidempt.loc[index, 'Date'],
                                         'SRRI': dfbothkiidempt.loc[index, 'SRRI'],
                                         'Source': 'KIID'}
        amount += 1
        rawdata.loc[amount] = {'SecId': dfbothkiidempt.loc[index, 'SecId'],
                                         'Date': dfbothkiidempt.loc[index, 'Date'],
                                         'SRRI': dfbothkiidempt.loc[index, 'SRRI'],
                                         'Source': 'EMT'}
        amount += 1
        count += 1
    rawdata = rawdata.sort_values(by=['SecId', 'Date', 'Source']).reset_index()
    rawdata.drop(columns=['index'], inplace=True)
    logger.info('Insert source data for KIID & EMT end')

    logger.info('Try to get share class amount begin')
    # 获得Share Class 数量
    uniqueshareclass = pd.DataFrame(rawdata, columns=['SecId']).drop_duplicates()
    uniqueshareclassamount = len(uniqueshareclass)
    logger.info('There are {0} unique share classes'.format(uniqueshareclassamount))
    logger.info('Try to get share class amount end')

    # 获得包含KIID和EMT两个sources的share classes
    logger.info('Try to get multiple source share details begin')
    dfsharesourcegroup = rawdata.groupby(['SecId', 'Source']).size().reset_index()
    dfsharesourcegroup.columns = ['SecId', 'Source', 'amount']
    dfsharesourceresult = dfsharesourcegroup.groupby(['SecId']).size().reset_index()
    dfsharesourceresult.columns = ['SecId', 'amount']
    dfsharemultiplesourceresult = dfsharesourceresult[(dfsharesourceresult['amount'] > 1)]
    dfsharewithsinglesource = dfsharesourceresult[(dfsharesourceresult['amount'] == 1)]['SecId'].reset_index()
    dfsharewithsinglesource.drop(columns=['index'], inplace=True)
    shareamountwithmultiplesource = len(dfsharemultiplesourceresult)
    logger.info('There are {0} share classes with multiple sources'.format(shareamountwithmultiplesource))

    dfmultipledetails = rawdata[(rawdata['SecId'].isin(dfsharemultiplesourceresult['SecId'].values))].reset_index()
    dfmultipledetails.drop(columns=['index'], inplace=True)
    dfmultipledetails = dfmultipledetails.sort_values(by=['SecId', 'Date', 'Source'])
    dfmultiplegroupdetails = dfmultipledetails.set_index(keys=['SecId', 'Date'], append=False, drop=True)
    indexbymaxdate = dfmultiplegroupdetails.reset_index().groupby(['SecId'])['Date'].idxmax()
    dfmultiplewithmaxdate = dfmultipledetails.loc[indexbymaxdate]
    dfemtaslatest = dfmultiplewithmaxdate[(dfmultiplewithmaxdate['Source'] == 'EMT')]
    latestsharewithemtamount = len(dfemtaslatest)
    logger.info('The amount of latest multiple source shares with EMT is {0}'.format(latestsharewithemtamount))
    latestemtpercentage = latestsharewithemtamount / shareamountwithmultiplesource * 100
    logger.info('The percentage of latest multiple source shares with EMT is {0}'.format(latestemtpercentage))
    logger.info('Try to get multiple source share details end')

    dffrequency = calculatefrequency(dfmultipledetails)

    logger.info('Try to output excel begin')
    write = pd.ExcelWriter(outputpath)
    dfmultipledetails.to_excel(write,
                               sheet_name='sharewithmultiplesource',
                               index=False,
                               encoding='utf-8')
    dfsharewithsinglesource.to_excel(write,
                                     sheet_name='sharewithsinglesource',
                                     index=False,
                                     encoding='utf-8')
    dfemtaslatest.to_excel(write,
                           sheet_name='EMTAsLatest',
                           index=False,
                           encoding='utf-8')
    data = {'TotalShareAmount': [uniqueshareclassamount],
            'ShareAmountwithSingleSource': [len(dfsharewithsinglesource)],
            'ShareAmountwithMultipleSource': [shareamountwithmultiplesource],
            'LatestShareAmountWithEMT': [latestsharewithemtamount],
            'LatestShareWithEMTPercentage': [latestemtpercentage]}
    statisticssheet = pd.DataFrame(data)
    statisticssheet.to_excel(write,
                             sheet_name='statistics',
                             index=False,
                             encoding='utf-8')
    dffrequency.to_excel(write,
                         sheet_name='Frequency',
                         index=False,
                         encoding='utf-8')
    write.save()
    logger.info('Try to output excel end')


def calculatefrequency(dfmultipledetails):
    logger.info('Calculate for frequency begin')
    dffrequency = pd.DataFrame(columns=('ValueBox',
                                        'Scenario',
                                        'KIID_EMT_Amount',
                                        'KIID_EMT_Percent',
                                        'KIID_Amount',
                                        'KIID_Percent',
                                        'EMT_Amount',
                                        'EMT_Percent'))
    dffrequency['ValueBox'] = dffrequency['ValueBox'].apply(str)
    dffrequency['Scenario'] = dffrequency['Scenario'].apply(str)
    dffrequency['KIID_EMT_Amount'] = dffrequency['KIID_EMT_Amount'].apply(int)
    dffrequency['KIID_EMT_Percent'] = dffrequency['KIID_EMT_Percent'].apply(float)
    dffrequency['KIID_Amount'] = dffrequency['KIID_Amount'].apply(int)
    dffrequency['KIID_Percent'] = dffrequency['KIID_Percent'].apply(float)
    dffrequency['EMT_Amount'] = dffrequency['KIID_Amount'].apply(int)
    dffrequency['EMT_Percent'] = dffrequency['KIID_Percent'].apply(float)
    index = 0
    dffrequency.loc[index] = {'ValueBox': 'X<=31',
                              'Scenario': 'Monthly',
                              'KIID_EMT_Amount': 0,
                              'KIID_EMT_Percent': 0.0,
                              'KIID_Amount': 0,
                              'KIID_Percent': 0.0,
                              'EMT_Amount': 0,
                              'EMT_Percent': 0.0}
    index += 1
    dffrequency.loc[index] = {'ValueBox': '31<X<=62',
                              'Scenario': 'bi-Monthly',
                              'KIID_EMT_Amount': 0,
                              'KIID_EMT_Percent': 0.0,
                              'KIID_Amount': 0,
                              'KIID_Percent': 0.0,
                              'EMT_Amount': 0,
                              'EMT_Percent': 0.0}
    index += 1
    dffrequency.loc[index] = {'ValueBox': '62<X<=93',
                              'Scenario': 'Quarterly',
                              'KIID_EMT_Amount': 0,
                              'KIID_EMT_Percent': 0.0,
                              'KIID_Amount': 0,
                              'KIID_Percent': 0.0,
                              'EMT_Amount': 0,
                              'EMT_Percent': 0.0}
    index += 1
    dffrequency.loc[index] = {'ValueBox': '93<X<=186',
                              'Scenario': 'semi-annually',
                              'KIID_EMT_Amount': 0,
                              'KIID_EMT_Percent': 0.0,
                              'KIID_Amount': 0,
                              'KIID_Percent': 0.0,
                              'EMT_Amount': 0,
                              'EMT_Percent': 0.0}
    index += 1
    dffrequency.loc[index] = {'ValueBox': '186<X<=365',
                              'Scenario': 'annually',
                              'KIID_EMT_Amount': 0,
                              'KIID_EMT_Percent': 0.0,
                              'KIID_Amount': 0,
                              'KIID_Percent': 0.0,
                              'EMT_Amount': 0,
                              'EMT_Percent': 0.0}
    index += 1
    dffrequency.loc[index] = {'ValueBox': 'X>365',
                              'Scenario': 'unknown',
                              'KIID_EMT_Amount': 0,
                              'KIID_EMT_Percent': 0.0,
                              'KIID_Amount': 0,
                              'KIID_Percent': 0.0,
                              'EMT_Amount': 0,
                              'EMT_Percent': 0.0}
    shareclasslist = dfmultipledetails['SecId'].drop_duplicates().values
    logger.info('Need calculate: {0} share classes begin'.format(len(shareclasslist)))
    for index, shareclass in enumerate(shareclasslist):
        logger.info('Calculate the: {0} share'.format(index + 1))
        temp = dfmultipledetails[(dfmultipledetails['SecId'] == shareclass)]
        setfrequencyvalue(dffrequency, temp, 'KIID', 'KIID_Amount')
        setfrequencyvalue(dffrequency, temp, 'EMT', 'EMT_Amount')
        setfrequencyvalue(dffrequency, temp, 'KIID & EMT', 'KIID_EMT_Amount')

    logger.info('Calculate: {0} share classes end'.format(len(shareclasslist)))
    calculatescenariopercentage(dffrequency, len(shareclasslist))

    logger.info('Calculate for frequency end')
    return dffrequency


MONTHLY = 0
BIMONTHLY = 1
QUARTERLY = 2
SEMIANNUALLY = 3
ANNUALLY = 4
UNKNOWN = 5
def setfrequencyvalue(dffrequency, dfsharedetail, source, sourcecolumn):
    dfsharebysource = dfsharedetail[(dfsharedetail['Source'] == source)].reset_index()
    if len(dfsharebysource) > 0:
        if len(dfsharebysource) == 1:
            dffrequency.loc[UNKNOWN, sourcecolumn] += 1
        else:
            result = getfrequencycategory(dfsharebysource)
            dffrequency.loc[result, sourcecolumn] += 1


def getfrequencycategory(dfsharebysource):
    dfsharebysource = dfsharebysource.sort_values(by=['Date'], ascending=False).reset_index()
    daysum = 0
    for index, row in dfsharebysource.iterrows():
        if index < len(dfsharebysource) - 1:
            daysum += (dfsharebysource.loc[index, 'Date'] - dfsharebysource.loc[index + 1, 'Date']).days
    divided = (len(dfsharebysource) - 1)
    daymean = 366
    if divided > 0:
        daymean = daysum / divided
    if daymean <= 31:
        result = MONTHLY
    elif 31 < daymean <= 62:
        result = BIMONTHLY
    elif 62 < daymean <= 93:
        result = QUARTERLY
    elif 93 < daymean <= 186:
        result = SEMIANNUALLY
    elif 186 < daymean <= 365:
        result = ANNUALLY
    else:
        result = UNKNOWN
    return result


def calculatescenariopercentage(dffrequency, shareclassamount):
    dffrequency['KIID_EMT_Percent'] = (dffrequency['KIID_EMT_Amount'] / shareclassamount) * 100
    dffrequency['KIID_Percent'] = (dffrequency['KIID_Amount'] / shareclassamount) * 100
    dffrequency['EMT_Percent'] = (dffrequency['EMT_Amount'] / shareclassamount) * 100
    dffrequency['KIID_EMT_Percent'] = dffrequency['KIID_EMT_Percent'].apply(lambda x: round(x, 2))
    dffrequency['KIID_Percent'] = dffrequency['KIID_Percent'].apply(lambda x: round(x, 2))
    dffrequency['EMT_Percent'] = dffrequency['EMT_Percent'].apply(lambda x: round(x, 2))


def drawplotforresult():
    import matplotlib.pyplot as plt
    filepath = './output/staticticsresult.xlsx'
    dffrequency = pd.read_excel(filepath, sheet_name='Frequency')
    name_list = dffrequency['Scenario'].values
    colors = ['r', 'b', 'g', 'yellow', 'k', 'c',  'm', 'lime', 'pink', 'peru']
    x = list(range(len(dffrequency)))

    count = 0
    for column in dffrequency.columns:
        if 'Amount' in column:
            count += 1
    total_width, n = 0.8, count
    width = total_width / n

    colorindex = 0
    for column in dffrequency.columns:
        if 'Amount' in column:
            plt.bar(x, dffrequency[column],
                    width=width,
                    label=column,
                    tick_label=name_list,
                    fc=colors[colorindex])
            for i in range(len(x)):
                x[i] = x[i] + width
            colorindex += 1

    plt.legend()
    plt.show()

if __name__ == '__main__':
    # startjob()
    drawplotforresult()