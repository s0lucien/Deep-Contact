import numpy as np


# ref: Lee - 2010 - Solving the Shallow Water equations using 2D SPH
def W_poly6_2D(r, h):
    '''
    :param r: A 2d matrix where each column is a vector
    :param h: Support radius
    :return:
    '''
    assert r.shape[0] == 2

    # We determine the length of the vectors and pick out those with length below h
    r_norm = np.linalg.norm(r, axis=0)
    W = np.zeros(r_norm.shape)
    W_i = np.where(r_norm < h)

    # We determine the weights
    c = 4 / (np.pi * np.power(h, 8))
    h2 = np.power(h, 2)
    W[W_i] = c * np.power(h2 - np.power(r_norm[W_i], 2), 3)

    # We normalize the weights so that they add up to 1
    W_sum = np.sum(W)
    W = W / W_sum

    return W

def spiky_2D(r, h):
    assert r.shape[0] == 2

    r_norm = np.linalg.norm(r, axis=0)
    W = np.zeros(r_norm.shape)
    W_i = np.where(r_norm < h)

    c = 15 / (np.pi * np.power(h, 6))
    h2 = np.power(h, 2)
    W[W_i] = c * np.power(h - np.abs(r_norm[W_i]), 3)

    # We normalize the weights so that they add up to 1
    W_sum = np.sum(W)
    W = W / W_sum

    return W
# no need to implement these yet
# def nablaW_poly6_2D(r, h):
#     pass
#
#
# def nabla2W_poly6_2D(r, h):
#     pass

# ref : https://www.youtube.com/watch?v=SQPCXzqH610
def W_poly6_3D(r, h):
    assert r.shape[0]==3
    r_norm = np.linalg.norm(r,axis=0)
    W = np.zeros(r_norm.shape)
    W_i = np.where(h >= r_norm)[0]
    W[W_i] = 315 / (64 * np.pi * np.power(h, 9)) * \
                np.power((np.power(h, 2) - np.power(r_norm[W_i], 2)), 3)
    return W


def nablaW_poly6_3D(r, h):
    assert r.shape[0] == 3
    r_norm = np.linalg.norm(r,axis=0)
    nablaW = np.zeros(r.shape)
    W_i = np.where(h >= r_norm)[0]
    r_unit = (r/r_norm)[:,W_i]
    nablaW[:,W_i] = -45 / (np.pi * np.power(h, 6)) * \
            np.power((np.power(h, 2) - np.power(r_norm[W_i], 2)), 2) * r_unit
    return nablaW


def nabla2W_poly6_3D(r, h):
    assert r.shape[0] == 3
    r_norm = np.linalg.norm(r,axis=0)
    nabla2W = np.zeros(r_norm.shape)
    W_i = np.where(h >= r_norm)[0]
    nabla2W[W_i] = 45 / (np.pi * np.power(h, 6)) * \
                    (h - r_norm[W_i])
    return nabla2W


# We may consider trying these out as well
# def W_spiky(r, h):
#     pass
#
#
# def nablaW_spiky(r, h):
#     pass
#
#
# def nabla2W_spiky(r, h):
#     pass
#
#
# def W_m4(r, h):
#     pass
#
#
# def nablaW_m4(delta_r, h):
#     pass
#
#
# def nabla2W_m4(delta_r, h):
#     pass
