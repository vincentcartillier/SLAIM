import os
import sys
import numpy as np

# -- linking Instant-NGP here
sys.path.append("dependencies/instant-ngp/build/")
import pyngp as ngp # noqa


def bound_to_vertices(mcb):
    #import pdb; pdb.set_trace()
    vertices = []
    minb = mcb[:,0]
    maxb = mcb[:,1]
    tmp = minb.copy()
    vertices.append(tmp.copy())
    tmp[0] = maxb[0]
    vertices.append(tmp.copy())
    tmp[1] = maxb[1]
    vertices.append(tmp.copy())
    tmp[2] = maxb[2]
    vertices.append(tmp.copy())
    tmp[1] = minb[1]
    vertices.append(tmp.copy())
    tmp[0] = minb[0]
    vertices.append(tmp.copy())
    tmp[1] = maxb[1]
    vertices.append(tmp.copy())
    tmp[2] = minb[2]
    vertices.append(tmp.copy())
    return vertices

# -- some tool functions
# -- taken from NGP
def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2,1e-2,3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]],[v[2], 0, -v[0]],[-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))


def convert_marching_cubes_bound_to_NGP(cfg):
    mcb = cfg.DATASET.MARCHING_CUBES_BOUND
    mcb = np.array(mcb) # [3,2]
    vertices = bound_to_vertices(mcb)
    vertices = np.array(vertices)
    vertices = np.transpose(vertices)

    # adjustment given poses pre-process
    # -- swap y and z
    vertices = vertices[[1,0,2],:] #swap y and z
    # -- flip world upside down
    vertices[2,:] *= -1 # flip whole world upside down

    up = np.array(cfg.DATASET.POSES_UP_VECTOR)
    R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
    R = np.pad(R,[0,1])
    R[-1, -1] = 1
    vertices = np.pad(vertices,((0,1), (0,0)))
    vertices[3,:] = 1
    vertices =  np.matmul(R, vertices)
    vertices = vertices[:3,:]

    # -- center point of attention
    totp = np.array(cfg.DATASET.POSES_CENTER_POINT)
    vertices -= totp[:,np.newaxis]

    # -- pose scale
    poses_scale = cfg.DATASET.POSES_SCALE
    vertices *= poses_scale

    # adjustment to NGP space
    ngp_scale = cfg.RENDERER.SCALE
    ngp_offset = np.array(cfg.RENDERER.OFFSET)
    vertices *= ngp_scale
    vertices += ngp_offset[:,np.newaxis]
    vertices[[0,1,2],:] = vertices[[1,2,0],:]

    # - get aabb bound
    mcb_min = np.min(vertices, axis=1)
    mcb_max = np.max(vertices, axis=1)

    bound = ngp.BoundingBox(mcb_min, mcb_max)
    return bound

















