from scipy.special import fresnel
import numpy as np
import matplotlib.pyplot as plt

class EulerRing(object):
    def __init__(self, x=0, y=0, circumference=1, num=400, endpoint=False):
        t = np.linspace(0, 1, num//4)
        s,c = fresnel(t)
        
        # shift (c,s) down to lower right quadrant
        s = s-s[-1]
        
        # (c,s) is now lower right quadrant of euler ring with circumference 4*1 = 4.
        #   scale to requested circumference:
        s = s/4 * circumference
        c = c/4 * circumference

        # build ring out of four quadrants
        c_rev = c[::-1]
        s_rev = s[::-1]
        self.x = x + np.concatenate((c[:-1],  c_rev[:-1], -c[:-1], -c_rev[:-1]))
        self.y = y + np.concatenate((s[:-1], -s_rev[:-1], -s[:-1],  s_rev[:-1]))
        
        if endpoint:
            self.x = np.append(self.x, self.x[0])
            self.y = np.append(self.y, self.y[0])
