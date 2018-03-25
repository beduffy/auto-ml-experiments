import matplotlib.pylab as plt
import numpy as np

print('np.sin(np.pi/2.):', np.sin(np.pi/2.))
print('np.sin(np.pi/4) :', np.sin(np.pi/4.))
print('np.sin(np.pi)   :', np.sin(np.pi))
print('np.sin(-np.pi)  :', np.sin(-np.pi))
print('np.sin(0)  :', np.sin(0))
print('np.sin(1)  :', np.sin(1))

x = np.linspace(-np.pi, np.pi, 201)
plt.plot(x, np.sin(x))
plt.xlabel('Angle [rad]')
plt.ylabel('sin(x)')
plt.axis('tight')
plt.show()

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable as ag

# ds = [str(x) for x in list(model.named_modules())]
# for i in ds:
#     print('\n type(m) == {}. = '.format(type(i), i))
#     print('\n type(m) == {}. = '.format(type(i[0]), i[0]))
#     print('\n type(m) == {}. = '.format(type(i[0][0]), i[0][0]))
#     print('\n type(m) == {}. = '.format(type(i[0][0][0]), i[0][0][0]))
