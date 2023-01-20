# -*- coding: utf-8 -*-
# ------------------------
# - Utilities script -
# ------------------------


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import date
import pandas_datareader as web
import yfinance as yfin
import datetime as dt
import sys
import os

plt.style.use("fivethirtyeight")
# Define dates to start and end
initial_stock_date = dt.datetime.now().date() - dt.timedelta(days=3*365)
final_stock_date = dt.datetime.now().date()

# Define a ação para procurar
STOCK_NAME = 'BOVA11.SA'