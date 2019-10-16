import numpy as np
import scipy

def spline_even_knots(num_ctl_points, order):
    knots_len = num_ctl_points + order + 1
    return order*[0] + list(np.linspace(0, 1, knots_len-2*order)) + order*[1]

def spline_curvature(spline, t):
    d1 = spline.derivative(1)(t)
    d2 = spline.derivative(2)(t)
    num = d1[:,0]*d2[:,1] - d1[:,1]*d2[:,0]
    denom = np.power(np.sum(d1*d1, axis=1), 3/2)
    denom[denom==0.0] = 1e-6
    return (num.T/denom).T

def bode_knee(x, knee=1, width=0.1):
    j = 0+1j
    e = np.exp(1)
    y = np.log((j*np.exp(x/width) - np.exp(knee/width)))*width - knee
    y[np.isinf(y)] = x[np.isinf(y)]
    y[np.isnan(y)] = x[np.isnan(y)]
    return y.real

def curvature_cost(spline, min_radius=0.1):
    N = 1000
    max_curvature = 1/min_radius
    t = np.linspace(0, 1, N)
    curv = np.abs(spline_curvature(spline, t))
    #curv_rect = 1-emopt.fomutils.rect(curv, 2*max_curvature, max_curvature/10)
    curv_knee = bode_knee(curv, max_curvature)
    cost = np.sum(curv_knee) / N
    return cost

def intersection_cost(spline1, spline2, min_width=0.2):
    N = 1000
    t = np.linspace(0, 1, N)
    points1 = spline1(t)
    points2 = spline2(t)
    dist_matrix = scipy.spatial.distance.cdist(points1, points2, 'euclidean')
    print(f'rank={RANK} minimum distance = {np.min(dist_matrix)}')
    inv_dist_matrix = 1.0 / dist_matrix;
    inv_dist_bode = bode_knee(1.0/dist_matrix, 1.0/min_width)
    cost = np.sum(inv_dist_bode) / (N*N)
    return 1000 * cost

