#TODO build model
#TODO caompile model
#TODO train the model
#TODO evaluate model
#TODO make predictions

'''Example of how gonna performance the model
input --> array([[0.1,0.2],[0.2,0.2]])
output --> array([[0.3],[0.4]])
'''

# import libraries
import numpy as np
from random import random


x = np.array([[random()/2 for _ in range(2)] for _ in range(2000)])