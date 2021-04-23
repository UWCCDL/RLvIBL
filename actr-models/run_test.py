import run_fitting as fit
import os
import time
import pandas as pd


HCPIDs = ["102311_fnca", "105115_fnca", "108323_fnca", "100307_fnca"]

min_output = {}

for HCPID in HCPIDs:
	t0 = time.time()
	print('########## start: ', HCPID, '##########')

	m1_min = fit.estimate_param_model(HCPID, "model1")
	t1  = time.time()
	print( HCPID, 'model1 DONE')
	
	m2_min = fit.estimate_param_model(HCPID, "model2")

	t2 =  time.time()
	
	min_output[HCPID] = [str(t1-t0), str(m1_min), str(t2-t1), str(m2_min)]
	print( HCPID, 'model2 DONE')
	
df = pd.DataFrame.from_dict(min_output, orient='index')
df.to_csv('./model_output/'+pd.to_datetime('now').strftime('%Y%m%d')+'_optimization_log.csv')

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