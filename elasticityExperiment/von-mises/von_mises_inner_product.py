#!/usr/bin/env python
# -*- coding: utf-8 -*-
#温刚
#北京大学数学科学学院，jnwengang@pku.edu.cn

import numpy as np
mu, kappa =1.7, 7.0 # mean and dispersion
s = np.random.vonmises(mu, kappa, 10000)
import matplotlib.pyplot as plt
from scipy.special import i0
plt.hist(np.cos(s), 50, density=True)
# plt.hist(s, 50, density=True)
x = np.linspace(-np.pi, np.pi, num=51)
y = np.exp(kappa*np.cos(x-mu))/(2*np.pi*i0(kappa))
plt.plot(x, y, linewidth=2, color='r')
plt.show()