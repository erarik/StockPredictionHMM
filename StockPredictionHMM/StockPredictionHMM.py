
import os

import numpy as np
import pandas as pd
import time
from datetime import timedelta
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import datetime

class StockDb(object):
    """ Stock Qutotes DB
    This class has been designed to provide a convenient interface for individual stock data.
    For example, to instantiate and load train/test files using a feature_method 
	definition named features, the following snippet may be used:
        stock = StockDb()
        stock.build_training(tr_file, features)
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
        """Creates sequence data objects for training stocks suitable for hmmlearn library
        :param feature_list: list of str label names
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


def print_model_stats(stock, model):
    print("Number of states trained in model for {} is {}".format(stock, model.n_components))    
    variance=np.array([np.diag(model.covars_[i]) for i in range(model.n_components)])    
    for i in range(model.n_components):
        print("hidden state #{}".format(i))
        print("mean = ", model.means_[i])
        print("variance = ", variance[i])
        


def train_a_stock(key, num_hidden_states):

    stock= StockDb()
    stock.df['fracchange'] = (stock.df['close'] - stock.df['open'])/stock.df['open']
    stock.df['fraclow'] = (stock.df['low'] - stock.df['open'])/stock.df['open']
    stock.df['frachigh'] = (stock.df['high'] - stock.df['open'])/stock.df['open']
    stock.df['frachighlow'] = (stock.df['high'] - stock.df['low'])/stock.df['low']
    stock.df['delta-volume']= stock.df['volume'].diff().fillna(0)
    stock.df['delta-open']= stock.df['open'].diff().fillna(0)
    stock.df['delta-close']= stock.df['close'].diff().fillna(0)
    stock.df['delta-closeopen']= stock.df['open'] - stock.df['close'].shift(-1).fillna(0)

    startdate=datetime.datetime(2018,3,6,0,0)
    enddate=datetime.datetime(2018,4,2,0,0)
    nbpredict = 40

    features=['delta-closeopen', 'fracchange', 'delta-volume']
    features_predict=['date','open', 'low', 'high', 'close']

    predict_df = pd.DataFrame(columns=features_predict)
    for i in range(nbpredict):
        try:
            train_df = stock.build_training(key, startdate, enddate, features)
            X, lengths = train_df.seq_len_dict(key)
            model = GaussianHMM(n_components=num_hidden_states, n_iter=1000).fit(X, lengths)
            logL = model.score(X, lengths)
            state_sequence = model.predict(X, lengths)
            prob_next_step = model.transmat_[state_sequence[-1], :]
            state_most_probable = state_sequence[0]
            state_feaures = model.means_[state_most_probable]
                        
            result={}
            result['date']=train_df.nextdata['date'].iloc[-2]
            result['logL']=logL
            result['nbState']=model.n_components
            result['state_most_probable']=state_most_probable
            result['prob_state_most_probable']=max(prob_next_step)

            result['open']=train_df.nextdata['close'].iloc[-1] +  state_feaures[features.index('delta-closeopen')]
            result['close']=result['open'] * (1 + state_feaures[features.index('fracchange')])
            result['fracchange'] = state_feaures[features.index('fracchange')]
            result['delta-closeopen'] = state_feaures[features.index('delta-closeopen')]
            predict_df = predict_df.append(result,  ignore_index=True)
        except:
            print("err")

        enddate = train_df.nextdata['date'].iloc[-2]

    real_df = stock.df[stock.df['symbol']==key]
    res_df = pd.merge(real_df, predict_df, on='date')
    res_df['err']=(res_df['fracchange_x']-res_df['fracchange_y'])/res_df['fracchange_x']
    print("----------------------------------------")

    # ========================
    # Plot the data
    # ========================
    fig, axes = plt.subplots(nrows=1, ncols=2)
     
    res_df.plot(kind='line', color='Blue', x='date', y='fracchange_x', ax=axes[0], title='fracchange')
    res_df.plot(kind='line', color='red', x='date', y='fracchange_y', ax=axes[0])
     
    res_df.plot(kind='line', color='Blue', x='date', y='err', ax=axes[1], title='fraclow')
    #res_df.plot(kind='line', color='red', x='date', y='fraclow_y', ax=axes[1])
     
    plt.show()

    print(res_df[['date','fracchange_x','fracchange_y']])

    return result








if __name__ == '__main__':
    stock= StockDb()
    stock.df['ratio'] = (stock.df['close'] - stock.df['open'])/stock.df['open']
    stock.df['ratio2'] = (stock.df['high'] - stock.df['low'])/stock.df['low']
    print(stock.df.head())
    startdate=datetime.datetime(2018,3,6,0,0)
    enddate=datetime.datetime(2018,4,2,0,0)
    train_df = stock.build_training('AMZN', startdate, enddate, ['volume', 'ratio', 'ratio2'])
    print("Training words: {}".format(train_df.stocks))

    train_a_stock('AMZN', 5)