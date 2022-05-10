from tensorflow import keras
from keras import datasets
from numpy import array
from numpy.linalg import norm
import random
import IISL_FLpkg.data_generator as dg
import IISL_FLpkg.model_generator as mg

N = 30

sca_metric = keras.metrics.SparseCategoricalAccuracy(name="sca")
prox_sca_metric = keras.metrics.SparseCategoricalAccuracy(name="prox_sca")
all_models, central_server = mg.model_generation(10, sca_metric, 0)
prox_all_models, prox_central_server = mg.model_generation(10, prox_sca_metric, 0)
X, Y = dg.generate_synthetic(1, 1, N, 60, 10, 0)

loss_list = []
accuracy_list = []
prox_loss_list = []
prox_accuracy_list = []

x = [[] for _ in range(10)]
y = [[] for _ in range(10)]

for iter in range(200):
    random_numbers = random.sample(range(29),10)
    for i in range(10):
        index = random_numbers[i]
        length = len(X[index])/10 - 1
        sel = random.randint(0, int(length))
        x[i] = X[index][sel*10:(sel+1)*10]
        y[i] = Y[index][sel*10:(sel+1)*10]
           
    results = all_models.fed_avg(x, y, sca_metric, central_server)
    prox_results = prox_all_models.fed_prox(x, y, prox_sca_metric, prox_central_server)
    loss_list.append(results[0])
    accuracy_list.append(results[1])
    prox_loss_list.append(prox_results[0])
    prox_accuracy_list.append(prox_results[1])
    if((iter+1)%10==0):
        print("[0]loss : %.10f, sca : %.10f" %(results[0], results[1]))
        print("[1]loss : %.10f, sca : %.10f" %(prox_results[0], prox_results[1]))
        print()