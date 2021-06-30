###################### ACT-R + PYTHON TEMPLATE #######################
#   Author: Cher Yang
#   Date: 5.20.2021
# This script is a python version of jupyter notebook demo
#
# Bugs: 
#
# TODO: 5.20 downsampling 
# 		6.8 fix the downsampling issue
# 			try mode as inputs
# Requirement: 
#
#
#
###################### ####################### #######################

# Futures
from __future__ import print_function
from random import randint
import warnings

from scipy.stats.stats import mode
warnings.filterwarnings('ignore')

# Built-in/Generic Imports
import os, glob
import sys 
import itertools 
import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns
from lasso_func import *
from dmd_func import *

sns.set_style("whitegrid")

class LassoAnalysis:
	def __init__(self, task_dir='/REST1/', ses_dir='/ses-01/'):
		self.power2011=None
		self.model_dat=None
		self.subj_dat=None
		
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

		# define model
		#self.model=LogisticRegression(penalty='l1', solver='saga', fit_intercept=False, max_iter=10000, tol=0.01)
		
		
	def loading(self, task_dir, ses_dir, fname='raw_pcorr.txt'):
		assert(fname in ['raw_corr_pearson.txt', 'raw_pcorr.txt', 'mr_corr_pearson.txt', 'mr_corr_spearman.txt', 'mr_pcorr.txt', 'g_dmdcorr.csv'])
		"""load model data, subject matrix, and power parcellation labels"""
		print('Loading  ... \n\t{}\n\t{}\n\t{}\n'.format(task_dir, ses_dir, fname))

		power2011=pd.read_csv('../bin/power_2011.csv', usecols=["ROI", "X", "Y", "Z", "Network", "Color", "NetworkName"]) 
		model_dat=pd.read_csv('../actr-models/model_output/MODELLogLikelihood.csv', index_col=0)
		model_dat['best_model1'] = np.where(model_dat['best_model']== 'm1', 1, 0)
		
		# raw_corr_pearson.txt
		# raw_pcorr.txt
		# mr_corr_pearson.txt
		# mr_corr_spearman.txt
		# mr_pcorr.txt

		if fname=='g_dmdcorr.csv':
			dmd_corr_df=pd.read_csv('./bin/{}_{}_{}'.format(task_dir.strip('/'), ses_dir.strip('/'), fname))
			subj_dat=pd.merge(left=model_dat[['HCPID', 'best_model1']], right=dmd_corr_df, how='right', on='HCPID')
		else:
			subj_dat=load_subj(model_dat, CORR_DIR='./connectivity_matrix', TASK_DIR=task_dir,
			SES_DIR=ses_dir, corr_fname=fname, znorm=True, warn=False)
		
		self.power2011=power2011
		self.model_dat=model_dat
		self.subj_dat=subj_dat	

	def preprocessing(self):
		# upsampling -> downsampling
		#subj_balanced=balance_training_sample(self.subj_dat, self.DV, method='up')	
		
		# only keep half of predictors	(69,696/2)	
		censor=get_vector_df(self.power2011, self.NOI)  
		subj_censored=get_subj_df(self.subj_dat, censor)	

		# feastures
		features = list(subj_censored.columns)[2:]
		
		#self.subj_balanced=subj_balanced
		self.subj_censored=subj_censored 
		self.features = features

	def load_param_grid(self):
		param_grid={}
		if self.model.__class__.__name__=='LogisticRegression':
			param_grid['C']=1/np.logspace(-3, 3, 100)
		elif self.model.__class__.__name__=='LinearSVC':
			param_grid['C']=1/np.logspace(-3, 3, 100)
		elif self.model.__class__.__name__=='RandomForestClassifier':
			param_grid={
				'max_depth':list(range(1,15))}
		elif self.model.__class__.__name__=='DecisionTreeClassifier':
			param_grid = {
				'max_depth':list(range(1,15))}
		else:
			print('Wrong Model Name!')
		self.param_grid = param_grid
	
	def set_cache_prefix(self, prefix_str):
		self.cache_prefix=prefix_str

	def set_balancing_type(self, type):
		assert type in ['up', 'down', 'none', 'balanced']
		
		if type=='up':
			balanced = balance_training_sample(self.subj_censored, self.DV, method='up')
			self.X=balanced[self.features]
			self.y=balanced[self.DV]
		elif type=='down':
			balanced = balance_training_sample(self.subj_censored, self.DV, method='down')	
			self.X=balanced[self.features]
			self.y=balanced[self.DV]
		elif type=='balanced':
			self.model.set_params(class_weight='balanced')
			self.X=self.subj_censored[self.features]
			self.y=self.subj_censored[self.DV]	
		else:
			self.model.set_params(class_weight=None)
			self.X=self.subj_censored[self.features]
			self.y=self.subj_censored[self.DV]	
		
	def hypertuning_lambda(self, load_lambda=None, method='standard_gs'):
		#lambda_range = 1.0/np.logspace(-3, 3, 100)
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
			coefs = save_regularization_path(self.subj_censored[self.features], self.subj_censored[self.DV], lambda_values = 1.0/np.logspace(-3, 3, 50), best_lambda=self.best_lambda)
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
		subj_mean_mat=average_corr(self.subj_dat, self.DV)
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
		plt.rcParams['xtick.labelsize'] = 20
		plt.rcParams['ytick.labelsize'] = 20
		plot_brain_connections(self.w_mat, self.power_coords, mat_name='beta_mat', thre='99.9%', save_plot=save_plot, cache_prefix=self.cache_prefix)

	def plot_heatmaps(self, save_plot=False):
		sns.heatmap(self.beta_mat, center=0, robust=True, xticklabels=False, yticklabels=False, cmap='coolwarm')
		plt.title('Beta Matrix')
		if save_plot: plt.savefig('./bin/'+self.cache_prefix+'beta_mat.png')
		plt.show()
		plt.close()

		sns.heatmap(self.subj_mean_mat, center=0, vmax=1, vmin=-1, xticklabels=False, yticklabels=False,  cmap='coolwarm')
		plt.title('A Matrix')
		if save_plot: plt.savefig('./bin/'+self.cache_prefix+'A_mat.png')
		plt.show()
		plt.close()

		sns.heatmap(self.w_mat, robust=True, xticklabels=False, yticklabels=False, cmap='coolwarm')
		plt.title('W Matrix')
		if save_plot: plt.savefig('./bin/'+self.cache_prefix+'W_mat.png')
		plt.show()
		plt.close()

	
def loadLasso():	
	#r1s1: lambda=6.73415065775082, score=0.936363636363636
	#r1s2: 1.232846739442066
	#r2s1: 0.93260334688322

	# init
	R1S1Lasso=LassoAnalysis()
	R1S1Lasso.loading(task_dir='/REST1/', ses_dir='/ses-01/')
	R1S1Lasso.preprocessing()
	R1S1Lasso.set_cache_prefix('r1s1_')

	# find lambda
	R1S1Lasso.hypertuning_lambda(load_lambda=6.73415065775082)
	R1S1Lasso.regularization_performance(load_data=True)
	R1S1Lasso.logistic_model_performance(load_data=True)

	# calculate B, A, W, matrix
	R1S1Lasso.logistic_model_retraining()
	R1S1Lasso.calc_weighted_corr()
	
	# plotting
	R1S1Lasso.plot_regularization(save_plot=True)
	R1S1Lasso.plot_performance(save_plot=True)
	R1S1Lasso.plot_brain_connectivity(save_plot=True)
	
	return R1S1Lasso

def runLasso():	
	# init
	R1S1Lasso=LassoAnalysis()
	R1S1Lasso.loading(task_dir='/REST1/', ses_dir='/ses-01/')
	R1S1Lasso.preprocessing()
	R1S1Lasso.set_cache_prefix('r1s1_')

	# find optimal lambda
	R1S1Lasso.hypertuning_lambda()
	R1S1Lasso.regularization_performance()

	# evaluate model
	R1S1Lasso.logistic_model_performance()

	# calculate B, A W matrix
	R1S1Lasso.logistic_model_retraining()
	R1S1Lasso.calc_weighted_corr()
	
	# plotting
	R1S1Lasso.plot_regularization(save_plot=True)
	R1S1Lasso.plot_performance(save_plot=True)
	R1S1Lasso.plot_brain_connectivity(save_plot=True)
	
	return R1S1Lasso

def runComparison():
	
	comparison_dict={'task_type':['/REST1/', '/REST2/'], 'ses_type':['/ses-01/', '/ses-02/'],
						'input_type':['raw_pcorr.txt', 'mr_corr_pearson.txt', 'g_dmdcorr.csv'], 
						'balance_type':['up', 'down', 'none', 'balanced'], 
						'model_type':[LogisticRegression(penalty='l1', solver='saga', fit_intercept=False, max_iter=10000, tol=0.01),
										
										DecisionTreeClassifier(),
										RandomForestClassifier(bootstrap=True, max_features='auto')]}
	keys, values = zip(*comparison_dict.items())
	comparison_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

	#comparison_df = pd.DataFrame(comparison_list)
	#comparison_df.to_csv('./bin/comparison_log.csv')
	
	for i in range(len(comparison_list)):
		c=comparison_list[i] # curr list
		
		#check exists
		pref='{task_type}_{ses_type}_{input_type}_{balance_type}_{model_type}_'.format(
							task_type=c['task_type'].split('/')[1],
							ses_type=c['ses_type'].split('/')[1],
							input_type=c['input_type'].split('.')[0].split('_')[1],
							balance_type=c['balance_type'], 
							model_type=c['model_type'].__class__.__name__)
		fname1='./bin/'+pref+'hyperparam_score.csv'
		fname2='./bin/'+pref+'evaluation_score.csv'
		if (os.path.exists(fname1) & os.path.exists(fname2)):
			print('Skipping...\n\t{}\n\t{}'.format(fname1, fname2))
		else:
			A=LassoAnalysis()
			A.loading(task_dir=c['task_type'], ses_dir=c['ses_type'], fname=c['input_type'])
			A.preprocessing()

			A.model=c['model_type']
			A.load_param_grid()
			A.set_balancing_type(c['balance_type'])
			A.set_cache_prefix(pref)
			# hyper tunning
			grid_search = tune_hyperparam(A.model, A.X, A.y, A.param_grid, cv=20)
			grid_search_results = pd.DataFrame(grid_search.cv_results_)
			grid_search_results.to_csv('./bin/'+A.cache_prefix+'hyperparam_score.csv')
			if (c['model_type'].__class__.__name__=='LogisticRegression'):
				plot_hyperparam(grid_search_results, save_path='./bin/'+A.cache_prefix+'hyperparam_score.png')

			# evaluate model
			A.best_model = grid_search.best_estimator_
			eval_scores = evaluate_model(A.best_model, A.X, A.y, cv=A.y.value_counts()[0])
			eval_scores.to_csv('./bin/'+A.cache_prefix+'evaluation_score.csv')

			# log 
			c['gs_best_params_'] = grid_search.best_params_
			c['gs_best_score_'] = grid_search.best_score_
			c['evaluation_accuracy'] = eval_scores['accuracy'].mean()
			c['evaluation_auc'] = eval_scores['roc_auc'].mean()
			pd.DataFrame.from_dict(c, orient='index').T.to_csv('./bin/comparison_log.csv', mode='a', header=not(i))
 
def main():
	runLasso()

if __name__ == "__main__":	
	main()
