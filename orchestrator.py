import os
import subprocess

param1_name = "z_size"
param2_name = "wasserstein_weight"
param3_name = "data_set"

param1= [64, 128]
param2= [10, 20, 30]
param3 = ["[sketchrnn_cat.npz]", "[sketchrnn_tractor.npz]", "[sketchrnn_cat.npz,sketchrnn_tractor.npz]"]

experiment = 1
num_steps = 80000

for p1 in param1:
	for p2 in param2:
		for p3 in param3:
			params = '--hparams=num_steps=' + str(num_steps) + "," + param1_name + "=" + str(p1) + "," + param2_name + "=" + str(p2) + "," + param3_name + "=" + str(p3)
			subprocess.call(["python3", "sketch_rnn_train.py", "--log_root=models/"+str(experiment), "--data_dir=dataset/quickdraw", params])
			experiment = experiment + 1


