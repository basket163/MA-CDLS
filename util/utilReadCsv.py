import util

import pandas as pd
import numpy as np
import xlrd

def readCsv(filePath):
    df = pd.read_csv(filePath)
    return df

def readExcel2Matrix(filePath):
    table = xlrd.open_workbook(filePath).sheets()[0]
    row = table.nrows
    col = table.ncols
    datamatrix = np.zeros((row,col))
    for x in range(col):
        cols = np.matrix(table.col_values(x))
        datamatrix[:, x] = cols
    #print(datamatrix)
    return datamatrix

def getCellDistance(matrix,cellx,celly):
    row = cellx - 1
    col = celly - 1
    value = matrix[row,col]
    return value