import numpy as np
from mindquantum import Circuit, RY, RX, RZ
from mindquantum import X, Z, Y


from mindquantum.simulator import Simulator
from mindquantum.core import ParameterResolver
from mindquantum.core.parameterresolver import ParameterResolver as PR
import math

class Parameter_manager:
    def __init__(self, key='default'):
        self.parameters = []
        self.count = 0
        self.key = key
        self.grad_key = None
    
    def init_parameter_resolver(self):
        pr = {k:np.random.randn()*2*math.pi for k in self.parameters}
        # pr = {k:0 for k in self.parameters}
        pr = ParameterResolver(pr)
        return pr

    def _replay(self):
        self.count = 0

    def set_grad_key(self, key):
        self.grad_key = key
        self._replay()    

    def create(self):
        param = '{}_theta_{}'.format(self.key, self.count)
        self.count += 1
        self.parameters.append(param)
        if self.grad_key is None or param!=self.grad_key:
            is_grad = False
        else:
            is_grad = True
        return param, is_grad


def RY_gate(circ, i, P):
    ry, is_grad = P.create()
    if not is_grad:
        circ += RY(ry).on(i)
    else:
        circ += Y.on(i)
        circ += RY(ry).on(i)

def RX_gate(circ, i, P):
    rx, is_grad = P.create()
    if not is_grad:
        circ += RX(rx).on(i)
    else:
        circ += X.on(i)
        circ += RX(rx).on(i)

def RZ_gate(circ, i, P):
    rz, is_grad = P.create()
    if not is_grad:
        circ += RZ(rz).on(i)
    else:
        circ += Z.on(i)
        circ += RZ(rz).on(i)

def RZZ_gate(circ, i, j, P):
    circ += X.on(j, i)
    RZ_gate(circ, j, P)
    circ += X.on(j, i)

def RYY_gate(circ, i, j, P):
    circ += X.on(j, i)
    RY_gate(circ, j, P)
    circ += X.on(j, i)

def RXX_gate(circ, i, j, P):
    circ += X.on(j, i)
    RX_gate(circ, j, P)
    circ += X.on(j, i)


def layer(C, P, n_qubits):
    for i in range(n_qubits):
        RX_gate(C, i, P)

    for i in range(0, n_qubits-1, 2):
        RZZ_gate(C, i, i+1, P)


if __name__ == '__main__':
    n_qubits = 3
    P = Parameter_manager()
    
    circ = Circuit()
    P = Parameter_manager()
    layer(circ, P, n_qubits)
    print(circ)

    circ = Circuit()
    P.set_grad_key('default_theta_0')
    layer(circ, P, n_qubits)
    print(circ)