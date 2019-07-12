import numpy as np
import neuroptica as neu
import scipy

def build_network_linear(L=2, N=4):
    layers = []
    for i in range(0, L):
        layers.append(neu.ClementsLayer(N))
    
    layers.append(neu.Activation(neu.Abs(N)))
    layers.append(neu.DropMask(N, keep_ports=[0]))
    
    print("Created %d layer LINEAR network" % L )
    
    return neu.Sequential(layers)


def build_network(g=np.pi*0.65, g_taper=1.0, phi_b=0.0, L=2, N=4, Nout=1, alpha=0.1):
    layers = []
    for i in range(0, L):
        eo_settings = { 'alpha': alpha, 'g': g*(g_taper**i), 'phi_b': phi_b }
        layers.append(neu.ClementsLayer(N))
        layers.append(neu.Activation(neu.ElectroOpticActivation(N, **eo_settings)))
    
    layers.append(neu.Activation(neu.Abs(N)))
#     layers.append(neu.Activation(neu.SoftMax(N)))
    layers.append(neu.DropMask(N, keep_ports=[0]))
    
    print("Created %d layer network" % L )
    
    return neu.Sequential(layers)


def generate_data_XOR(N, y_scale=1.0, normalize=False):
    assert N <= 8
    Y = np.zeros( (1, 2**N) )
    number_array = np.array( [np.array([i]) for i in range(0,2**N)], dtype=np.uint8 )
    bit_array    = np.unpackbits(number_array, axis=1)
    X = bit_array.astype(np.complex)[:, (8-N):].T
    inds = np.count_nonzero(X, axis=0) == 1
    Y[0,inds] = y_scale
    X = X + 1e-9
    if normalize:
        X = X/np.linalg.norm(X, axis=0)
        X[:,0] = 1e-9
        
    return X, Y

