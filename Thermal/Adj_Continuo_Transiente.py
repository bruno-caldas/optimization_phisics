#-*- coding: utf-8 -*-
# Otimizacao de Sistema Termico Transiente com Sensibilidades Calculadas pelo Dolfin Adjoint
# Script criado por:
#       Diego Silva Prado           prado.dis@gmail.com
#       Bruno Caldas de Souza

from dolfin import *
from ROL.dolfin_vector import DolfinVector as FeVector
import numpy
import ROL

set_log_level(30)
CRED = '\033[91m'
CYELLOW = '\33[33m'
CEND = '\033[0m'
CGREEN  = '\33[32m'
CBEIGE  = '\33[36m'
CBOLD     = '\33[1m'
CVIOLET = '\33[35m'

print(" ╔════════════════════════════════════════════════════════════════════════════════════════════╗")
print(" ║"+ CGREEN + CBOLD + " Otimizacao de Sistema Termico Transiente com Sensibilidades Calculadas pelo Dolfin Adjoint "+CEND+"║")
print(" ║"+ CGREEN + " Script criado por Diego Silva Prado e Bruno Caldas de Souza                                "+CEND+"║")
print(" ╚════════════════════════════════════════════════════════════════════════════════════════════╝")

pasta = "continuo_transiente/"

#Propriedades do material
rho = Constant(1.)
cp = Constant(1.)

#Parametros do modelo de material
eps = Constant(1.0e-8)
p = Constant(5.)
alpha = Constant(1.0e-8)
beta = Constant(1.0e0)
V = Constant(0.25)

#Transiente
dt = 0.2
tfin = 5.

#Condicoes de Contorno
Tpoint = 3000. #Temperatura no centro
Tamb = 273. #Temperatura ambiente

#Resolucao da malha
N = 60

#Mapa do estado inicial de temperatura
class Heatpoint(Expression):
    "Experimental - Porosity Map"
    def eval(self, value, x):
        material = 0
        if (x[0]-0.5)*(x[0]-0.5)+(x[1]-0.5)*(x[1]-0.5)<0.05*0.05:
            material = 1
        value[0] = Tpoint if material == 1 else Tamb

#Modelo de material para a conducao
def k(m):
    return (eps + (1 - eps) * m**p)*2.5

def kdash(m):
    return ( (1 - eps) * (p)* m**(p-1))*2.5

#Malha e espacos de funcao
mesh = RectangleMesh(mpi_comm_world(), Point(0.0, 0.0), Point(1.0, 1.0), N, N, 'crossed') # m
A = FunctionSpace(mesh, "DG", 0)        # control function space
T = FunctionSpace(mesh, "CG", 1)

#Variavel de projeto
m = interpolate(Constant(0.5), A)

#Problema direto
def prob_direto(m, t, t0):
    v = TestFunction(t.function_space())
    F = inner(-rho*cp*(t-t0)/dt,v)*dx - inner(k(m)*grad(t), grad(v))*dx
    bc1 = DirichletBC(T, 273., "on_boundary")
    return (F, bc1)

def Stsreturn(time):
    n = 70
    compn = int((time/tfin)*n)
    sts = ''
    for a in range(int(compn)):
        sts = sts+'█'
    for a in range(int(n-compn)):
        sts = sts+'░'
    return sts

#Solucao do problema direto
def resp_direto(m,nome):
    print(" ╔════════════════════════════════════════════════════════════════════════════════════════════╗")
    print(" ║"+ CYELLOW +"                          Calculando solucao do problema direto                             "+CEND+"║")
    print(" ╚════════════════════════════════════════════════════════════════════════════════════════════╝")
    t = Function(T)
    t0 = Function(T)
    t0 = interpolate(Heatpoint(degree=1),T)
    ts = 0.
    memoT = []
    memoT0 = []
    file_t = File(pasta + nome + ".pvd")
    while tfin+.0000000001 > ts:
        (F, bc1) = prob_direto(m, t, t0)
        stststr = Stsreturn(ts)
        print(' ║'+CRED+stststr+CEND+'║  Time: %.1f s / '+ str(tfin) + 's\r'.format(ts)) % ts,
        solve(F==0, t, [bc1])
        file_t << t
        if ts > tfin:
            memoT.append(t)
            memoT0.append(t0)
        t0.assign(t)
        ts += dt
    print ''
    print(CGREEN + ' CONCLUIDO ' + CEND)
    return t, memoT, memoT0

def functional(m, t):
    return ((t*t+beta*((m**2)*(1-m)**2))*dx)

def solve_adjoint(m, t, t0, J):
    (F, bc1) = prob_direto(m, t, t0)
    bc1.homogenize() # adjoint has homogeneous BCs
    adj = Function(T)
    adF = adjoint(derivative(F, t))
    dJ = derivative(J, t, TestFunction(t.function_space()))
    solve(action(adF, adj) - dJ == 0, adj, [bc1])
    #print("O adj eh %f " % float(adj[0]))
    print(adj)
    return adj

class L2Inner(object):

    def __init__(self):
        self.A = assemble(TrialFunction(A)*TestFunction(A)*dx)

    def eval(self, _u, _v):
        _v.apply('insert')
        A_u = _v.copy()
        self.A.mult(_u, A_u)
        return _v.inner(A_u)

    def riesz_map(self, derivative):
        res = Function(A)
        rhs = Function(A, derivative)
        solve(self.A, res.vector(), rhs.vector())
        return res.vector()

dot_product = L2Inner()

#Codigo principal
#Criando variavel: Temperatura
#t = Function(T)
#t, memoT = resp_direto(m,"problema_direto_inicial")

#Limites e restricao de volume
lb = 0.0
ub = 1.0

state_file = File(pasta + "state.pvd")
control_file = File(pasta + "control.pvd")

#Configurando Otimizador ROL
class ObjR(ROL.Objective):
    '''Subclass of ROL.Objective to define value and gradient for problem'''
    def __init__(self, inner_product):
        ROL.Objective.__init__(self)
        print(CGREEN + " ╔════════════════════════════════════════════════════════════════════════════════════════════╗" + CEND)
        print(CGREEN + " ║       Executando a Otimizacao com Sensibilidades Continuas (Calculadas no FENICS)          ║" + CEND)
        print(CGREEN + " ╚════════════════════════════════════════════════════════════════════════════════════════════╝" + CEND)
        self.inner_product = inner_product
        self.m = Function(A, name="Controle_Continuo")
        self.t = Function(T)
        self.t0 = Function(T)

    def value(self, x, tol):
        J = functional(self.m, self.t)
        return assemble(J)

    def gradient(self, g, x, tol):
        m = self.m
        t = self.t
        t0 = self.t0
        lam = solve_adjoint(m, t, t0, functional(m, t))
        dmo = TestFunction(A)
        L = beta*(2*m-(6*m**2)+(4*m**3)) * dmo * dx - kdash(m) * dmo * inner(grad(t), grad(lam) ) * dx #inner(k(m)*grad(t), grad(v))*dx
        deriv = assemble(L)
        if self.inner_product is not None:
            grad_ = self.inner_product.riesz_map(deriv)
        else:
            grad_ = deriv
        g.scale(0)
        g.vec += grad_
        self.deriva = grad_

    def update(self, x, flag, iteration):
        print('\n' + CBEIGE + '█████████████████████████████████████ !!! ATUALIZANDO !!! █████████████████████████████████████' + CEND)
        m = Function(A, x.vec, name="Controle_Continuo")
        self.m.assign(m)
        t, listaT, listaT0 = resp_direto(self.m,"atualizacao")
        self.t.assign(listaT[0])
        self.t0.assign(listaT0[0])
        if iteration >= 0:
            control_file << self.m
            state_file << self.t

    def resposta(self):
        return self.m


class VolConstraint(ROL.Constraint):

    def __init__(self, inner_product):
        ROL.Constraint.__init__(self)
        self.inner_product = inner_product

    def value(self, cvec, xvec, tol):
        a = Function(A, xvec.vec)
        val = assemble(a * dx) - V
        cvec[0] = val

    def applyJacobian(self, jv, v, x, tol):
        da = Function(A, v.vec)
        jv[0] = assemble(da * dx)

    def applyAdjointJacobian(self, ajv, v, x, tol):
        da = TestFunction(A)
        deriv = assemble(da*dx)
        if self.inner_product is not None:
            grad = self.inner_product.riesz_map(deriv)
        else:
            grad = deriv
        ajv.scale(0)
        ajv.vec += grad
        ajv.scale(v[0])

# Initialise 'ROLVector'
l = ROL.StdVector(1)
c = ROL.StdVector(1)
v = ROL.StdVector(1)
v[0] = 1.0
dualv = ROL.StdVector(1)
v.checkVector(c, l)

x = interpolate(Constant(0.5), A)
x = FeVector(x.vector(), dot_product)
g = Function(A)
g = FeVector(g.vector(), dot_product)
d = interpolate(Expression("1 + x[0] * (1-x[0])*x[1] * (1-x[1])", degree=1), A)
d = FeVector(d.vector(), dot_product)
x.checkVector(d, g)

jd = Function(A)
jd = FeVector(jd.vector(), dot_product)

lower = interpolate(Constant(0.0), A)
lower = FeVector(lower.vector(), dot_product)
upper = interpolate(Constant(1.0), A)
upper = FeVector(upper.vector(), dot_product)

# Instantiate Objective class for poisson problem
obj = ObjR(dot_product)
# obj.checkGradient(x, d, 4, 2)
volConstr = VolConstraint(dot_product)
volConstr.checkApplyJacobian(x, d, jd, 3, 1)
volConstr.checkAdjointConsistencyJacobian(v, d, x)

set_log_level(30)

paramsDict = {
        'General': {
            'Secant': { 'Type': 'Limited-Memory BFGS', 'Maximum Storage': 25 } },
            'Step': {
                'Type': 'Augmented Lagrangian',
                'Line Search': {
                    'Descent Method': {
                      'Type': 'Quasi-Newton Step'}},
                'Augmented Lagrangian': {
                    'Initial Penalty Parameter'               : 1.e2,
                    'Penalty Parameter Growth Factor'         : 2,
                    'Minimum Penalty Parameter Reciprocal'    : 0.1,
                    'Initial Optimality Tolerance'            : 1.0,
                    'Optimality Tolerance Update Exponent'    : 1.0,
                    'Optimality Tolerance Decrease Exponent'  : 1.0,
                    'Initial Feasibility Tolerance'           : 1.0,
                    'Feasibility Tolerance Update Exponent'   : 0.1,
                    'Feasibility Tolerance Decrease Exponent' : 0.9,
                    'Print Intermediate Optimization History' : True,
                    'Subproblem Step Type'                    : 'Line Search',
                    'Subproblem Iteration Limit'              : 10
                  }},
        'Status Test': {
            'Gradient Tolerance': 1e-15, 'Relative Gradient Tolerance': 1e-10,
            'Step Tolerance': 1e-16, 'Relative Step Tolerance': 1e-10,
            'Iteration Limit': 7}
        }
params = ROL.ParameterList(paramsDict, "Parameters")
bound_constraint = ROL.Bounds(lower, upper, 1.0)

optimProblem = ROL.OptimizationProblem(obj, x, bnd=bound_constraint, econ=volConstr, emul=l)
solver = ROL.OptimizationSolver(optimProblem, params)
solver.solve()

File(pasta + "Variavel_projeto_cont.pvd") << obj.resposta()
print CBEIGE + " ╔════════════════════════════════════════════════════════════════════════════════════════════╗"
print " ║              ♫                     Analise Concluida                         ♫             ║"
print " ╚════════════════════════════════════════════════════════════════════════════════════════════╝"+ CEND



#File(pasta + "Variavel_projeto_cont.pvd") << obj.resposta()
