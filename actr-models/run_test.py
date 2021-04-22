import run_fitting as fit
import os
import time
import pandas as pd


HCPIDs = ["102311_fnca", "105115_fnca", "108323_fnca", "100307_fnca"]

min_output = {}

for HCPID in HCPIDs:
	t0 = time.time()
	print('########## start: ', HCPID, '##########')

	m1_min = fit.estimate_param_model1(HCPID)
	t1  = time.time()
	print( HCPID, 'model1 DONE')
	
	m2_min = fit.estimate_param_model2(HCPID)
	t2 =  time.time()
	
	min_output[HCPID] = [str(t1-t0), str(m1_min), str(t2-t1), str(m2_min)]
	print( HCPID, 'model2 DONE')
	
df = pd.DataFrame.from_dict(min_output, orient='index')
df.to_csv('./model_output/min_log.csv')

