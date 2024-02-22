import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings(module='sklearn*', action='ignore', category=FutureWarning)
warnings.filterwarnings(module='imblearn*', action='ignore', category=UserWarning)
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

root_dir = '/home2/glee/Hyundai/MSSP/'

## Define class for data loading and preprocessing
class Battery:
    def __init__(self, path='../data/', method='method_total'):
        self.src = path+'carbattery_'+method+'.txt'

        # load raw data
        self.df = pd.read_csv(self.src, sep="\t")
        # data normalisation
        target = self.df.iloc[:,1:]
        self.df_norm = target.divide(target.iloc[0,:])
        self.df_norm.index = self.df['Cycle']

    # function to do data smoothing
    def smoothing(self, target_cycle=700):
        print("data smoothing...")
        for col in tqdm(self.df_norm.columns):
            sample = self.df_norm[col]
            for cy in range(3, target_cycle-3):
                if sample.loc[cy-2] >= sample.loc[cy-1] and sample.loc[cy-1] < sample.loc[cy] and sample.loc[cy] >= sample.loc[cy+1]:
                    sample.loc[cy] = (sample.loc[cy-1]+sample.loc[cy+1]) / 2

        for col in tqdm(self.df_norm.columns):
            sample = self.df_norm[col]
            for cy in range(3, target_cycle-3):
                if sample.loc[cy-2] <= sample.loc[cy-1] and sample.loc[cy-1] > sample.loc[cy] and sample.loc[cy] <= sample.loc[cy+1]:
                    sample.loc[cy] = (sample.loc[cy-1]+sample.loc[cy+1]) / 2
        print("done!")

    def smoothing2(self, target_cycle=700):
        print("data smoothing...")
        x = np.array(self.df_norm.index)[:target_cycle]
        for col in tqdm(self.df_norm.columns):
            self.df_norm[col][:target_cycle][self.df_norm[col][:target_cycle].isna()] = 0
            y = np.array(self.df_norm[col])[:target_cycle]
            itp = interp1d(x, y, kind='linear')
            w, p = 3, 2
            y_ = savgol_filter(itp(x), w, p)
            self.df_norm[col].iloc[:target_cycle] = pd.Series(y_, index=x, name=col)
        print("done!")

    # function to make datasets (train and test) for cross validation
    def make_inputs(self, input_cycle, target_cycle, select_features=False):
        # take last values at the target cycle (e.g. 700 cycle)
        temp_val = self.last_values(self.df_norm, target_cycle)
        # take indexes of data that has values at target cycle
        temp_idx = temp_val[temp_val.notna()].index
        temp_df = self.slicing(self.df_norm[temp_idx].copy(deep=True), target_cycle)

        self.X = temp_df.T

        if select_features:
            self.X_sliced = self.feature_selection(self.slicing(temp_df, input_cycle))
        else:
            self.X_sliced = self.slicing(temp_df, input_cycle).T

        self.Y = temp_val[temp_val.notna()].copy(deep=True)
        self.Y_label_binary, self.Y_label_multi = self.Y.copy(deep=True), self.Y.copy(deep=True)

        # Class labeling
        self.Y_label_binary[self.Y<0.7] = 0
        self.Y_label_binary[self.Y>=0.7] = 1

        self.Y_label_multi[self.Y<0.7] = 0
        self.Y_label_multi[self.Y>=0.7] = 1
        self.Y_label_multi[self.Y>=0.75] = 2
        self.Y_label_multi[self.Y>=0.8] = 3
        self.Y_label_multi[self.Y>=0.85] = 4
        # self.Y_label_multi[self.Y>=0.9] = 5

        # Save indices
        self.indices_name = self.Y.index.copy(deep=True)
        self.indices_integer = pd.Index(np.arange(len(self.Y)))

    # function to select features from raw data
    def feature_selection(self, X_src):
        n_moves = 10
        input_cycle = len(X_src)
        # moving average
        avglist = ('avg'+(',avg'.join(str(x) for x in list(range(0,input_cycle-10))))).split(',')
        # first-order difference
        fodlist = ('fod'+(',fod'.join(str(x) for x in list(range(0,input_cycle-10))))).split(',')
        # moving variance
        varlist = ('var'+(',var'.join(str(x) for x in list(range(0,input_cycle-10))))).split(',')

        features = []
        for i in range(0,input_cycle-10):
            features.append(avglist[i])
            features.append(fodlist[i])
            features.append(varlist[i])
        features = np.array(features)
        X_input = pd.DataFrame(index=X_src.columns, columns=features)
        for i in range(0,input_cycle-10):
            temp_avg = X_src.iloc[i:(i+n_moves)].mean(axis=0)
            X_input['avg'+str(i)] = temp_avg
            temp_fod = abs(X_src.iloc[i:(i+n_moves+1)].diff(periods=-1).iloc[:n_moves]).mean()
            X_input['fod'+str(i)] = temp_fod
            temp_var = X_src.iloc[i:(i+n_moves)].var(axis=0)
            X_input['var'+str(i)] = temp_var
            #temp_trend = self.trend_feature(X_src.iloc[0:(i+n_moves)])
            #X_input['trend'+str(i)] = temp_trend

        return X_input

    # function to compute trend feature by doing linear regression
    def trend_feature(self, X_src):
        idxs = np.array(range(len(X_src))).reshape(-1,1)
        lr = LinearRegression()
        lr.fit(idxs, X_src)

        return lr.coef_.reshape(-1)

    # function to remove nan values from the dataset
    def nan_filter(self, X, input_cycle, target_cycle):
        temp_val = self.last_values(X, target_cycle)
        temp_idx = temp_val[temp_val.notna()].index
        temp_input = self.last_values(self.df_norm, input_cycle)
        input_idx = temp_input[temp_input.notna()].index
        df_out = self.slicing(copy(X[input_idx]), target_cycle)

        return df_out

    # function to do extrapolation
    def fill_values(self, df, start_cycle, final_cycle, step=3):
        # nan at 700 cycle but not nan at 100(or 150, 200, ...)
        final_val = self.last_values(df, final_cycle)
        final_index = final_val[final_val.isna()].index
        temp_val = self.last_values(df[final_index], start_cycle)
        temp_index = temp_val[temp_val.notna()].index
        self.index_extrapolated = temp_index
        # for each index
        for idx in tqdm(temp_index):
            target = df.loc[:,idx]
            # first index with nan (after start_cycle before final_cycle)
            last_index = target[target.isna()].index[0]
            delta = (target.loc[last_index-1] - target.loc[last_index-step]) / (step-1)
            for idx_fill in range(last_index, final_cycle+1):
                new_val = target.loc[idx_fill-1] + delta
                if new_val <= 0:
                    target.iloc[idx_fill-1:final_cycle] = 0
                    break
                else:
                    target.loc[idx_fill] = new_val
        return df

    # function to take last values
    def last_values(self, df, cycle):
        return df.iloc[cycle-1,:]

    # function to slice dataset
    def slicing(self, df, cycle):
        return df.iloc[:cycle,:]
