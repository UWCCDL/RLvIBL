import run_fitting as fit
import run_model as run
import os
import time
import pandas as pd
import glob

#################### Optimization ####################
def HCPID_list(exclude_list = None):
	subj_paths = glob.glob("../bin/gambling_trials/*.csv")
	HCPIDs = [sub.split("/")[-1].split(".")[0] for sub in subj_paths]
	HCPIDs.sort()
	if exclude_list:
		HCPIDs = [sub for sub in HCPIDs if sub not in exclude_list]
	return HCPIDs

# def reformat_optimal_output(HCPID, m1_min, m2_min, t0, t1, t2):
# 	min_output={}
# 	min_output["HCPID"] = [HCPID]
# 	min_output["log_date"] = [pd.to_datetime('now').strftime("%y-%m-%d")]

# 	min_output["model1_time"] = [t1-t0]
# 	min_output["model1_sucess"] = [m1_min.success]
# 	min_output["model1_best_ans"] = [m1_min.x[0]]
# 	min_output["model1_best_bll"] = [m1_min.x[1]]
# 	min_output["model1_best_lf"] = [m1_min.x[2]]
# 	min_output["model1_obj_value"] = [m1_min.fun]
# 	min_output["model1_details"] = [str(m1_min)]

# 	min_output["model2_time"] = [t2-t1]
# 	min_output["model2_sucess"] = [m2_min.success]
# 	min_output["model2_best_egs"] = [m2_min.x[0]]
# 	min_output["model2_best_alpha"] = [m2_min.x[1]]
# 	min_output["model2_best_r"] = [m2_min.x[2]]
# 	min_output["model2_obj_value"] = [m2_min.fun]
# 	min_output["model2_details"] = [str(m2_min)]
# 	df = pd.DataFrame.from_dict(min_output)
# 	return df

# def run_optimization(HCPIDs):
# 	for HCPID in HCPIDs:

# 		t0 = time.time()
# 		print('########## start: ', HCPID, '##########')

# 		m1_min = fit.estimate_param(HCPID, "model1")
# 		t1  = time.time()

# 		print( HCPID, 'model1 DONE')

# 		m2_min = fit.estimate_param(HCPID, "model2")

# 		t2 =  time.time()
# 		print( HCPID, 'model2 DONE')

# 		# log optimization output
# 		min_output = reformat_optimal_output(HCPID, m1_min, m2_min, t0, t1, t2)
# 		df = pd.DataFrame.from_dict(min_output)

# 		fpath = "./model_output/param_optimization_log.csv"
# 		df.to_csv(fpath, mode="a", header=not(os.path.exists(fpath)), index=False)

def reformat_optimal_output_m2(HCPID, m2_min, t1, t2):
	min_output={}
	min_output["HCPID"] = [HCPID]
	min_output["log_date"] = [pd.to_datetime('now').strftime("%y-%m-%d")]

	min_output["model2_time"] = [t2-t1]
	min_output["model2_sucess"] = [m2_min.success]
	min_output["model2_best_egs"] = [m2_min.x[0]]
	min_output["model2_best_alpha"] = [m2_min.x[1]]
	min_output["model2_best_r"] = [m2_min.x[2]]
	min_output["model2_obj_value"] = [m2_min.fun]
	min_output["model2_details"] = [str(m2_min)]
	df = pd.DataFrame.from_dict(min_output)
	return df

def run_optimization_m2(HCPIDs):
	for HCPID in HCPIDs:

		#t0 = time.time()
		print('########## start: ', HCPID, '##########')

		# m1_min = fit.estimate_param(HCPID, "model1")
		t1  = time.time()

		# print( HCPID, 'model1 DONE')

		m2_min = fit.estimate_param(HCPID, "model2")

		t2 =  time.time()
		print( HCPID, 'model2 DONE')

		# log optimization output
		min_output = reformat_optimal_output_m2(HCPID, m2_min, t1, t2)
		df = pd.DataFrame.from_dict(min_output)

		fpath = "./model_output/m2_param_optimization_log.csv"
		df.to_csv(fpath, mode="a", header=not(os.path.exists(fpath)), index=False)

#################### Re-Run Optimized Simulation ####################
def run_optimal_m2(HCPID, epoch=800):
	param_optimization_log = pd.read_csv("./model_output/param_optimization_log.csv", header=0)
	param_array = param_optimization_log[param_optimization_log['HCPID']==HCPID]['model2_best_egs'].to_list() +\
	param_optimization_log[param_optimization_log['HCPID']==HCPID]['model2_best_alpha'].to_list() +\
	param_optimization_log[param_optimization_log['HCPID']==HCPID]['model2_best_r'].to_list()
	param_set = dict(zip(['egs', 'alpha', 'r'], param_array))
	model_output  = run.simulate(epoch=epoch, model="model2", param_set=param_set, export=True, verbose=False,
				 file_suffix="_" + HCPID + "_best" + str(epoch), HCPID=HCPID)
	return model_output


#################### Main ####################
def main():
	# exclude list
	#optimization_done = ["100307_fnca", "100408_fnca", "101006_fnca", "101107_fnca", "101309_fnca", "101410_fnca"]
	optimization_done = ["100307_fnca"]

	# define HCPID
	HCPIDs = HCPID_list(optimization_done)
	# run_optimization_m2(HCPIDs)

	# given optimal parameter, re-run 800 simulation
	for HCPID in HCPIDs:
		#run_optimal_m2(HCPID=HCPID, epoch=800)
		fit.grid_search_estimate_param(HCPID, "model1", 50)
		fit.grid_search_estimate_param(HCPID, "model2", 50)
		print(">> grid-search 100 done")

if __name__ == "__main__":
    main()

# estimate_param_model("102311_fnca", "model1")
#>> [0.95508218 0.61949251 0.48553827] rmse = 0.24615907901410006
# Optimization terminated successfully.
#          Current function value: 20.240320
#          Iterations: 1
#          Function evaluations: 59
#  allvecs: [array([0.1, 0.1, 0.1]), array([0.95508218, 0.61949251, 0.48557161])]
#    direc: array([[1., 0., 0.],
#        [0., 1., 0.],
#        [0., 0., 1.]])
#      fun: 20.240319685919253
#  message: 'Optimization terminated successfully.'
#     nfev: 59
#      nit: 1
#   status: 0
#  success: True
#        x: array([0.95508218, 0.61949251, 0.48557161])



# estimate_param_model("102311_fnca", "model2")
# >> [2.11417055 0.37796725 3.09911423] rmse = 0.2522426366350969
# # Optimization terminated successfully.
# #          Current function value: 0.243263
# #          Iterations: 2
# #          Function evaluations: 125
# #  allvecs: [array([0.1, 0.1, 0.1]), array([2.69027216, 0.38347961, 1.41965393]), array([2.11417055, 0.37796725, 3.09914759])]
# #    direc: array([[1., 0., 0.],
# #        [0., 1., 0.],
# #        [0., 0., 1.]])
# #      fun: 0.2432632234530605
# #  message: 'Optimization terminated successfully.'
# #     nfev: 125
# #      nit: 2
# #   status: 0
# #  success: True
# #        x: array([2.11417055, 0.37796725, 3.09914759])
