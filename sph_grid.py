import scipy.interpolate
import functools
import numpy as np
import pandas as pd

def W_poly6_2D(r, h):
    '''
    :param r: A 2d matrix where each column is a vector
    :param h: Support radius
    :return:
    '''
    # We determine the length of the vectors and pick out those with length below h
    W = np.zeros(r.shape)
    W_i = np.where(r < h)

    # We determine the weights
    c = 4 / (np.pi * np.power(h, 8))
    h2 = np.power(h, 2)
    W[W_i] = c * np.power(h2 - np.power(r[W_i], 2), 3)

    # We normalize the weights so that they add up to 1
    W_sum = np.sum(W)
    W = W / W_sum

    return W

class Grid:
    def __init__(self, p_ll, p_ur, res):
        (xlo, ylo), (xhi, yhi) , (xRes,yRes) = p_ll,p_ur, res

        self.x = np.arange(xlo, xhi+xRes, xRes)
        self.y = np.arange(ylo, yhi+xRes, yRes)
        self.Y, self.X = np.mgrid[xlo:(xhi+xRes):xRes, ylo:(yhi+yRes):yRes]
        self.point_coefs=None

    # sets grid coefficients. Is idempotent, use clear_coefs
    def _dist_to(self, points, support_radius):
        if self.point_coefs is not None : return self.point_coefs 
        px, py = points
#         import pdb;pdb.set_trace()
        P_grid   = np.c_[self.X.ravel(), self.Y.ravel()]
        P_points = np.c_[px, py]
        # For each point we determine all grid nodes within radius h
        KDTree = scipy.spatial.cKDTree(P_grid)
        NNs = KDTree.query_ball_point(P_points, support_radius)
        point_coefs = []
        for i in range(NNs.shape[0]):
            idx = np.unravel_index(NNs[i],self.X.shape)
            pts_around = P_grid[NNs[i]]
            distances = np.linalg.norm(pts_around - P_points[i],axis=1) # center around the point
            point_coefs.append((idx,distances))
        self.point_coefs = point_coefs
        return self.point_coefs
    
    def splatter(self, positions, values, support_radius, kernel=W_poly6_2D):
        p_rs = self._dist_to(positions,support_radius)
        I = np.zeros(self.X.shape)
        for i,p_r in enumerate(p_rs):
            p_ixs, rs = p_r
            coefs=kernel(rs,support_radius)
            I[p_ixs]+=coefs*values[i]
        return I

    def collect(self, Z, positions):
        f = scipy.interpolate.RectBivariateSpline(self.x, self.y, Z)
        res = []
        # import pdb; pdb.set_trace()
        for pt in positions:
            res.append(f(pt[0],pt[1]).item())
        return res
    
    def clear_coefs(self):
        self.point_coefs = None

def dataframe_to_grid(G, df:pd.DataFrame,channel,support_radius):
    ps = (df['px'].values,df['py'].values)
    return G.splatter(positions=ps,values=df[channel].values,support_radius=support_radius)