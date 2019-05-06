
import os

import numpy as np
import pandas as pd
import time
from datetime import timedelta

class StockDb(object):
    """ Stock Qutotes DB
    This class has been designed to provide a convenient interface for individual stock data.
    For example, to instantiate and load train/test files using a feature_method 
	definition named features, the following snippet may be used:
        stock = StockDb()
        stock.build_training(tr_file, features)
        stock.build_test(tst_file, features)
    """

    def __init__(self,
                 stocks_fn=os.path.join('data', 'stocks.csv'),
                 ):
        """ loads STOCK database from csv files
        :param stocks_fn:
            filename of stocks with expected format:
                symbol
        Instance variables:
            df: pandas dataframe
            
                snippit example:
                         symbol	date	close	volume	open	high	low
                            AMZN	06/05/2018	1696.3500	4733275.0000	1672.9900	1699.0000	1670.0600
                            AMZN	06/04/2018	1665.2700	3178387.0000	1648.9000	1665.6800	1645.4900
                            AMZN	06/01/2018	1641.5400	3302466.0000	1637.0300	1646.7299	1635.0900

        """
        stocksdf=[]
        for index, row in pd.read_csv(stocks_fn).iterrows():
            symbol = row[0]
            quotes_fn=os.path.join('data', "{}.csv".format(symbol))
            df = pd.read_csv(quotes_fn, dtype={'close': float, 'open': float, 'volume':float, 'high':float, 'low':float}, parse_dates=['date']).assign(symbol = symbol)#assign to add symbol column
            stocksdf.append(df)
        self.df = pd.concat(stocksdf)


    def build_training(self, key, startdate, enddate, feature_list):
        """ wrapper creates sequence data objects for training stocks suitable for hmmlearn library
        :param feature_list: list of str label names
        :param csvfilename: str
        :return: WordsData object
            dictionary of lists of feature list sequence lists for each stock
                {'AMZN': [[[87, 225], [87, 225], ...], [[88, 219], [88, 219], ...]]]}
        """
        self._dict = {}
        new_sequence = []

        _df = self.df[(self.df['date'] >= startdate) & (self.df['date'] <= enddate) & (self.df['symbol']==key)]

        for index, data in _df.iterrows():
            sample = [float(data[f]) for f in feature_list]
            if len(sample) > 0:  
                new_sequence.append(sample)            
            
        if key in self._dict:
            self._dict[key].append(new_sequence) # list of sequences
        else:
            self._dict[key] = [new_sequence]         

        self.nextdata = self.df[(self.df['date'] >= currdate) & (self.df['symbol']==key)]

		#create hmmlearn data
        self.seq_len_dict = {}
        for key in self._dict:
            sequence_cat = []
            sequence_lengths = []
            for sequence in self._dict[key]:
                sequence_cat += sequence
                num_frames = len(sequence)
                sequence_lengths.append(num_frames)

            self.seq_len_dict[key] = np.array(sequence_cat), sequence_lengths

if __name__ == '__main__':
    stock= StockDb()
    stock.df['ratio'] = (stock.df['close'] - stock.df['open'])/stock.df['open']
    stock.df['ratio2'] = (stock.df['high'] - stock.df['low'])/stock.df['low']
    print(stock.df.head())
    startdate=datetime.datetime(2018,3,6,0,0)
    enddate=datetime.datetime(2018,4,2,0,0)
    train_df = stock.build_training('AMZN', startdate, enddate, ['volume', 'ratio', 'ratio2'])
    print("Training words: {}".format(train_df.stocks))

