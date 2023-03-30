import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
import matplotlib as plt
from matplotlib import pyplot
from numpy import genfromtxt
my_data = genfromtxt('speeddating.csv', delimiter=',')
print(my_data.tostring())
