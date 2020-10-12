import random
import pandas as pd
import numpy as np
from functools import partial
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from xgboost.sklearn import XGBRegressor

import matplotlib.pyplot as plt
import seaborn as sns





def generate_data():
    x,y= load_boston(return_X_y=True)
    x_train, x_val, y_train, y_val=train_test_split(x,y,test_size=.2)
    
    return(x_train, x_val, y_train, y_val)





def plot_result(y_val,y_upper,y_lower):
    fig = plt.figure(figsize=(12,6))
    x_point=range(y_val.shape[0])
    pic_data=pd.DataFrame({'y':y_val,'upper':y_upper,'lower':y_lower}).sort_values('y')
    plt.plot(x_point,pic_data['y'], 'g:')
    plt.plot(x_point, pic_data['upper'], 'k-')
    plt.plot(x_point, pic_data['lower'], 'k-')
    plt.fill(np.concatenate([x_point, x_point[::-1]]),
         np.concatenate([pic_data['upper'], pic_data['lower'][::-1]]),
         alpha=.5, fc='r', ec='None')





def predict_interval(x_train,y_train,x_val,model,alpha=.9,xgb=False):
    
    model.fit(x_train,y_train)
    y_pred=model.predict(x_val)
    
    if xgb :
        model.set_params(loss='quantile', quant_alpha=alpha)
    model.set_params(loss='quantile', alpha=alpha)
    model.fit(x_train,y_train)
    y_upper= model.predict(x_val) 
    
    if xgb :
        model.set_params(loss='quantile', quant_alpha=1-alpha)
    model.set_params(loss='quantile', alpha=1-alpha)
    model.fit(x_train,y_train)
    y_lower= model.predict(x_val)
    
    return(y_upper,y_lower)





class XGBQuantile(XGBRegressor):
  def __init__(self,quant_alpha=0.95,quant_delta = 1.0,quant_thres=1.0,quant_var =1.0,base_score=0.5, booster='gbtree', colsample_bylevel=1,
                colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
                n_jobs=1, nthread=None, objective='reg:linear', random_state=0,reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,silent=True, subsample=1):
    self.quant_alpha = quant_alpha
    self.quant_delta = quant_delta
    self.quant_thres = quant_thres
    self.quant_var = quant_var
    
    super().__init__(base_score=base_score, booster=booster, colsample_bylevel=colsample_bylevel,
       colsample_bytree=colsample_bytree, gamma=gamma, learning_rate=learning_rate, max_delta_step=max_delta_step,
       max_depth=max_depth, min_child_weight=min_child_weight, missing=missing, n_estimators=n_estimators,
       n_jobs= n_jobs, nthread=nthread, objective=objective, random_state=random_state,
       reg_alpha=reg_alpha, reg_lambda=reg_lambda, scale_pos_weight=scale_pos_weight, seed=seed,
       silent=silent, subsample=subsample)
    
    self.test = None
  
  def fit(self, X, y):
    super().set_params(objective=partial(XGBQuantile.quantile_loss,alpha = self.quant_alpha,delta = self.quant_delta,threshold = self.quant_thres,var = self.quant_var) )
    super().fit(X,y)
    return self
  
  def predict(self,X):
    return super().predict(X)
  
  def score(self, X, y):
    y_pred = super().predict(X)
    score = XGBQuantile.quantile_score(y, y_pred, self.quant_alpha)
    score = 1./score
    return score
      
  @staticmethod
  def quantile_loss(y_true,y_pred,alpha,delta,threshold,var):
    x = y_true - y_pred
    grad = (x<(alpha-1.0)*delta)*(1.0-alpha)-  ((x>=(alpha-1.0)*delta)& (x<alpha*delta) )*x/delta-alpha*(x>alpha*delta)
    hess = ((x>=(alpha-1.0)*delta)& (x<alpha*delta) )/delta 
 
#     grad = (np.abs(x)<threshold )*grad - (np.abs(x)>=threshold )*(2*np.random.randint(2, size=len(y_true)) -1.0)*var
#     hess = (np.abs(x)<threshold )*hess + (np.abs(x)>=threshold )
    return grad, hess
  
  @staticmethod
  def original_quantile_loss(y_true,y_pred,alpha,delta):
    x = y_true - y_pred
    grad = (x<(alpha-1.0)*delta)*(1.0-alpha)-((x>=(alpha-1.0)*delta)& (x<alpha*delta) )*x/delta-alpha*(x>alpha*delta)
    hess = ((x>=(alpha-1.0)*delta)& (x<alpha*delta) )/delta 
    return grad,hess

  
  @staticmethod
  def quantile_score(y_true, y_pred, alpha):
    score = XGBQuantile.quantile_cost(x=y_true-y_pred,alpha=alpha)
    score = np.sum(score)
    return score
  
  @staticmethod
  def quantile_cost(x, alpha):
    return (alpha-1.0)*x*(x<0)+alpha*x*(x>=0)
  
  @staticmethod
  def get_split_gain(gradient,hessian,l=1):
    split_gain = list()
    for i in range(gradient.shape[0]):
      split_gain.append(np.sum(gradient[:i])**2/(np.sum(hessian[:i])+l)+np.sum(gradient[i:])**2/(np.sum(hessian[i:])+l)-np.sum(gradient)**2/(np.sum(hessian)+l) )
    
    return np.array(split_gain)



