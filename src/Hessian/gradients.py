import mindquantum.core.gates as G
from mindquantum.simulator import Simulator
from mindquantum.core.operators import QubitOperator
from mindquantum import Hamiltonian
import copy
import numpy as np

from .utils import pr2array



def get_gradient_preconditional(gate):
    if isinstance(gate, G.RX):
        return G.X.on(gate.obj_qubits)
    elif isinstance(gate, G.RY):
        return G.Y.on(gate.obj_qubits)
    elif isinstance(gate, G.RZ):
        return G.Z.on(gate.obj_qubits)
    else:
        raise NotImplementedError()

J = np.complex(0,1)
def grad_circuit_symbolic_forward(circ):
    circ_list, circ_coeff_list = [], []
    for i, gate in enumerate(circ):
        if isinstance(gate, (G.RX, G.RY, G.RZ)):
            n_circ = copy.deepcopy(circ)
            n_circ.insert(i, get_gradient_preconditional(gate))
            circ_list.append(n_circ)
            circ_coeff_list.append(-1./2 * J) #for example, grad(RX) = -j X RX
    return circ_list, circ_coeff_list

class Grad:
    def __init__(self, circ, pr, ham, n_qubits):
        self.circ, self.pr, self.ham = circ, pr, ham
        self.n_qubits = n_qubits
        self.parameters, self.k_list = pr2array(self.pr)
        self.circ_list, self.circ_coeff_list = grad_circuit_symbolic_forward(self.circ)

        assert len(self.circ_list)==len(self.k_list), '{} vs {}'.format(len(self.circ_list), len(self.k_list))

        self.phase_shift = None
        
    def determine_phase_shift(self):
        self.phase_shift = -1
#         jac, _ = self._grad(1)
#         jac_reverse = self.grad_reserveMode()
#         self.phase_shift = jac.real.mean() / jac_reverse.real.mean()
#         self.pahse_shift = self.phase_shift / np.abs(self.phase_shift)
#         print('phase_shift', self.phase_shift)
#         indices = np.argsort(jac)[::-1]
#         print(jac[indices[:5]], jac_reverse[indices[:5]], jac.max(), jac.min(), jac_reverse.max(), jac_reverse.min())
        
    def grad(self):
        if self.phase_shift is None:
            self.determine_phase_shift()
        return self._grad(self.phase_shift)
        
    def _grad(self, phase_shift):
        r'''
        calculate gradient using forwardMode, while also calculate Hessian with hybridMode
        '''
        jac = np.zeros(len(self.parameters)).astype(np.complex)
        hess = np.zeros((len(self.parameters), len(self.parameters))).astype(np.complex)

        for i, (circ_right, coeff) in enumerate(zip(self.circ_list, self.circ_coeff_list)):
            sim = Simulator('projectq', self.n_qubits)
            circ_right.no_grad()
            grad_ops = sim.get_expectation_with_grad(self.ham, circ_right, self.circ)
            e, g = grad_ops(self.parameters)
            jac[i] = e[0][0] * coeff #this is \partial E/ \partial circ_right
            hess[i] = g.squeeze() * coeff *(-1) * phase_shift#* J

        jac = jac * 2 * phase_shift #+ jac * J # add h.c.

        return jac.real, hess.real

    def Hess_forwardMode(self):
        r'''
        calculate Hessian using forward mode
        '''
        if self.phase_shift is None:
            self.determine_phase_shift()
            
        hess = np.zeros((len(self.parameters), len(self.parameters))).astype(np.complex)
        for i, (circ_left, coeff_left) in enumerate(zip(self.circ_list, self.circ_coeff_list)):
            for j, (circ_right, coeff_right) in enumerate(zip(self.circ_list, self.circ_coeff_list)):
                sim = Simulator('projectq', self.n_qubits)
                circ_right.no_grad()
                circ_left.no_grad()

                grad_ops = sim.get_expectation_with_grad(self.ham, circ_right, circ_left)
                e, g = grad_ops(self.parameters)
                hess[i][j] = e[0][0] * coeff_left * coeff_right * self.phase_shift * self.phase_shift #* J

        return hess.real

    def grad_reserveMode(self):
        r'''
        test method that generate gradient using backpropogation(reverse mode differentiation)
        '''
        sim = Simulator('projectq', self.n_qubits)
        grad_ops = sim.get_expectation_with_grad(self.ham, self.circ, self.circ)
        e, g = grad_ops(self.parameters)
        return g.squeeze().real


class FisherInformation:
    def __init__(self, circ, pr, n_qubits):
        ham = QubitOperator('')
        ham = Hamiltonian(ham)
        self.G = Grad(circ, pr, ham, n_qubits)

        self.circ, self.pr, self.ham = circ, pr, ham
        self.n_qubits = n_qubits

    def gite_preconditional(self):
        jac, hess = self.G.grad()
        return hess

    def fisher_information(self):
        jac, hess = self.G.grad()
        jac_left = np.expand_dims(jac, axis=1)
        jac_right = np.expand_dims(jac, axis=0) * J

        matrix = hess - (jac_left * jac_right)
        matrix = 4 * matrix.real
        return matrix

    # def fisherInformation_builtin(self):
    #     sim = Simulator('projectq', self.n_qubits)
    #     # matrix = sim.get_fisher_information_matrix(circ, self.G.parameters, diagonal=False)
    #     matrix = sim.fisher_information_matrix(circ.get_cpp_obj(),
    #                                               circ.get_cpp_obj(hermitian=True),
    #                                               self.G.parameters,
    #                                               circ.params_name,
    #                                               False)
    #     return matrix

