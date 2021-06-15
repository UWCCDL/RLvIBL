###################### ACT-R + PYTHON TEMPLATE #######################
#   Author: Cher Yang
#   Date: 5.20.2021
# This script is a python version of jupyter notebook demo
#
# Bugs: 
#
# TODO: downsampling, use mode as input
# 
# Requirement: 
#
#
#
###################### ####################### #######################

# Futures
from __future__ import print_function 
import warnings
warnings.filterwarnings('ignore')

# Built-in/Generic Imports
import os, glob
import sys 
import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns
from lasso_func import *

sns.set_style("whitegrid")

class LassoAnalysis:
	def __init__(self, task_dir='/REST1/', ses_dir='/ses-01/', corr_fname='mr_pcorr.txt'):
		self.power2011=None
		self.model_dat=None
		self.subj_mat=None
		
		# L1 outputs
		self.best_lambda=None
		self.coefs=None
		self.scores=None
		
		self.cache_prefix=''		# cached output

		# default 
		self.DV='best_model1'
		self.NOI=[
		"Uncertain", 
		"Sensory/somatomotor Hand",
		"Sensory/somatomotor Mouth",
		"Cingulo-opercular Task Control",
		"Auditory",
		"Default mode",
		"Memory retrieval?",
		"Ventral attention",
		"Visual",
		"Fronto-parietal Task Control",
		"Salience",
		"Subcortical",
		"Cerebellar",
		"Dorsal attention"]
		
		# load
		self.loading(task_dir, ses_dir, corr_fname)
		# preprocess
		self.preprocessing()	
		
		
	def loading(self, task_dir, ses_dir, corr_fname):
		"""load model data, subject matrix, and power parcellation labels"""
		power2011=pd.read_csv('../bin/power_2011.csv', usecols=["ROI", "X", "Y", "Z", "Network", "Color", "NetworkName"]) 
		model_dat=pd.read_csv('../actr-models/model_output/MODELLogLikelihood.csv', index_col=0)
		model_dat['best_model1'] = np.where(model_dat['best_model']== 'm1', 1, 0)
		#subj_mat=load_subj(model_dat, task_dir, ses_dir, corr_fname, warn=False)
		subj_mat=load_subj(model_dat, CORR_DIR='./connectivity_matrix', TASK_DIR=task_dir,
		 SES_DIR=ses_dir, corr_fname='mr_pcorr.txt', znorm=True, warn=False)
		
		self.power2011=power2011
		self.model_dat=model_dat
		self.subj_mat=subj_mat		
					
	def preprocessing(self):
		# upsampling -> downsampling
		subj_balanced=balance_training_sample(self.subj_mat, self.DV)	
		
		# only keep half of matrix		
		censor=get_vector_df(self.power2011, self.NOI)  
		subj_censored=get_subj_df(subj_balanced, censor)	

		# feastures
		features = list(subj_censored.columns)[2:]
		
		self.subj_balanced=subj_balanced
		self.subj_censored=subj_censored 
		self.features = features
	
	def set_cache_prefix(self, prefix_str):
		self.cache_prefix=prefix_str
		
	def hypertuning_lambda(self, load_lambda=None, method='standard_gs', lambda_range = 1.0/np.logspace(-3, 3, 100)):
		if load_lambda!=None:
			self.best_lambda=load_lambda
			print('Loading best lambda', self.best_lambda)
			return None
		else:
			if method == 'standard_gs':
				grid_result = grid_search_lasso(self.subj_censored[self.features], self.subj_censored[self.DV])
			elif method == 'random_gs':
				grid_result = random_grid_search_lasso(self.subj_censored[self.features], self.subj_censored[self.DV])
		
			self.best_lambda = 1/grid_result.best_params_['C']
			print('Finding best Lambda :', self.best_lambda)
			return grid_result
	
	def regularization_performance(self, save_data=True, load_data=False):
		if load_data:
			coefs = pd.read_csv('./bin/'+self.cache_prefix+'coef_results.csv')
			scores = pd.read_csv('./bin/'+self.cache_prefix+'score_results.csv')
			self.coefs=coefs
			self.scores=scores
			print('Loading regularization results')
		else:
			coefs = save_regularization_path(self.subj_censored[self.features], self.subj_censored[self.DV], lambda_values = 1.0/np.logspace(-2, 2, 10), best_lambda=self.best_lambda)
			coefs = pd.DataFrame(coefs)
			scores = save_regularization_score(self.subj_censored[self.features], self.subj_censored[self.DV], 1.0/self.best_lambda, num_cv=20)
			
			self.coefs=coefs
			self.scores=scores
			if save_data: 
				coefs.to_csv('./bin/'+self.cache_prefix+'coef_results.csv', index=False)
				scores.to_csv('./bin/'+self.cache_prefix+'score_results.csv', index=False)
		
			
	def logistic_model_performance(self, save_data=True, load_data=False):
		if load_data: 
			pred_data=pd.read_csv('./bin/'+self.cache_prefix+'loo_results.csv')
			self.pred_data=pred_data
			print('Loading prediction results')
		else:
			pred_data=loocv_logistic_retrain(self.subj_censored, self.features, self.DV, 1/self.best_lambda)
			print('The Leave-One-Out Accuracy Score: {:.4f}'.format(accuracy_score(pred_data['ytrue'], pred_data['yhat'])))
			self.pred_data=pred_data
			if save_data: pred_data.to_csv('./bin/'+self.cache_prefix+'loo_results.csv', index=False)
		return pred_data
		
	def logistic_model_retraining(self):
		best_logit_L1=LogisticRegression(penalty='l1', solver='saga', C=1/self.best_lambda, fit_intercept=False)
		best_logit_L1.fit(self.subj_censored[self.features], self.subj_censored[self.DV])
		self.best_logit_L1=best_logit_L1
		return best_logit_L1 
		
	def calc_weighted_corr(self):	
		censored=get_vector_df(self.power2011, self.NOI)
		beta_mat, power_coords = map_beta(censored, self.best_logit_L1.coef_[0], self.power2011)
		
		# average corr matrix across subj
		# calcualte beta * averaged PR   
		subj_mean_mat=average_corr(self.subj_mat, self.DV)
		subj_mean_mat=pd.DataFrame(subj_mean_mat, columns=beta_mat.columns)
		w_mat=beta_mat * subj_mean_mat
		
		self.beta_mat=beta_mat
		self.power_coords=power_coords
		self.subj_mean_mat=subj_mean_mat
		self.w_mat=w_mat
	
	def plot_regularization(self, save_plot=False):
		plt.rcParams.update({'font.size': 40})
		plt.rcParams['lines.linewidth'] = 10
		plt.rcParams['xtick.labelsize'] = 30
		plt.rcParams['ytick.labelsize'] = 30
		
		plot_regularization_path(self.coefs, self.best_lambda, 1.0/np.logspace(-2, 2, 100), save_plot=save_plot, cache_prefix=self.cache_prefix) 
		plot_regularization_score(self.scores, self.best_lambda, save_plot=save_plot, cache_prefix=self.cache_prefix)
		return
		
	def plot_performance(self, save_plot=False):
		plt.rcParams.update({'font.size': 40})
		plt.rcParams['lines.linewidth'] = 10
		plt.rcParams['xtick.labelsize'] = 30
		plt.rcParams['ytick.labelsize'] = 30
		plot_confusion_matrix_loo(self.pred_data, save_plot=save_plot, cache_prefix=self.cache_prefix)
		plot_roc_curve_loo(self.pred_data, save_plot=save_plot, cache_prefix=self.cache_prefix)
		plot_prediction_loo(self.pred_data, threshold=0.5, drop_dup=False, save_plot=save_plot, cache_prefix=self.cache_prefix+'bal_')
		plot_prediction_loo(self.pred_data, threshold=0.5, drop_dup=True, save_plot=save_plot, cache_prefix=self.cache_prefix+'unbal_')
		return
		
	def plot_brain_connectivity(self, save_plot=False):
		plt.rcParams['lines.linewidth'] = 20
		plt.rcParams['xtick.labelsize'] = 30
		plt.rcParams['ytick.labelsize'] = 30
		plot_brain_connections(self.w_mat, self.power_coords, mat_name='beta_mat', thre='99.9%', save_plot=save_plot, cache_prefix=self.cache_prefix)

def loadLasso():	
	#r1s1: 0.93260334688322
	#r1s2: 1.232846739442066
	#r2s1: 0.93260334688322
	R1S1Lasso=LassoAnalysis(task_dir='/REST1/', ses_dir='/ses-01/')
	R1S1Lasso.set_cache_prefix('r1s1_')
	R1S1Lasso.hypertuning_lambda(load_lambda=0.93260334688322)
	R1S1Lasso.regularization_performance(load_data=True)
	R1S1Lasso.logistic_model_performance(load_data=True)
	R1S1Lasso.logistic_model_retraining()
	R1S1Lasso.calc_weighted_corr()
	
	# plotting
	R1S1Lasso.plot_regularization(save_plot=True)
	R1S1Lasso.plot_performance(save_plot=True)
	R1S1Lasso.plot_brain_connectivity(save_plot=True)
	
	return R1S1Lasso

def runLasso():	
	R1S1Lasso=LassoAnalysis(task_dir='/REST1/', ses_dir='/ses-01/')
	R1S1Lasso.set_cache_prefix('r1s1_')
	R1S1Lasso.hypertuning_lambda()
	R1S1Lasso.regularization_performance()
	R1S1Lasso.logistic_model_performance()
	R1S1Lasso.logistic_model_retraining()
	R1S1Lasso.calc_weighted_corr()
	
	# plotting
	R1S1Lasso.plot_regularization(save_plot=True)
	R1S1Lasso.plot_performance(save_plot=True)
	R1S1Lasso.plot_brain_connectivity(save_plot=True)
	
	return R1S2Lasso

def main():
	runLasso()

if __name__ == "__main__":	
	main()
