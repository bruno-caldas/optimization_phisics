from dolfin import *
from ROL.dolfin_vector import DolfinVector as FeVector
import numpy
import ROL

pasta = "continuo/"

rho = Constant(703.)
cp = Constant(1000.)
k = Constant(300.)
N = 50
eps = Constant(1.0e-3)
p = Constant(5.)
alpha = Constant(0.01)

def k(m):
    return (eps + (1 - eps) * m**p)#*10000.

def kdash(m):
    return ( (1 - eps) * (p)* m**(p-1))#*10000.

mesh = RectangleMesh(mpi_comm_world(), Point(0.0, 0.0), Point(1.0, 1.0), N, N, 'crossed') # m
A = FunctionSpace(mesh, "DG", 0)        # control function space
T = FunctionSpace(mesh, "CG", 1)

m = interpolate(Constant(0.5), A)
V = 0.3

def prob_direto(m, t):
    v = TestFunction(t.function_space())
    F = inner(k(m)*grad(t), grad(v))*dx
    bc1 = DirichletBC(T, 273., "on_boundary")
    bc2 = DirichletBC(T, 1000., "(x[0]-0.5)*(x[0]-0.5)+(x[1]-0.5)*(x[1]-0.5)<0.1*0.1")
    return (F, bc1, bc2)

def resp_direto(m):
    t = Function(T)
    (F, bc1, bc2) = prob_direto(m, t)
    solve(F==0, t, [bc1, bc2])
    File(pasta + "Resposta_Direto.pvd") << t
    return t

def solve_adjoint(m, t, J):
    (F, bc1, bc2) = prob_direto(m, t)
    bc1.homogenize() # adjoint has homogeneous BCs
    bc2.homogenize() # adjoint has homogeneous BCs
    adj = Function(T)

    adF = adjoint(derivative(F, t))
    dJ = derivative(J, t, TestFunction(t.function_space()))

    solve(action(adF, adj) - dJ == 0, adj, [bc1, bc2])
    #print("O adj eh %f " % float(adj[0]))
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


m_file = File(pasta + "Variavel_projeto.pvd")

lb = 0.0
ub = 1.0
#volume_constraint = UFLInequalityConstraint((0.3 - m)*dx, Control(m))
def functional(m, t):
    return (t*dx -inner( k(m), k(m) )*dx )

state_file = File(pasta + "state.pvd")
control_file = File(pasta + "control.pvd")
class ObjR(ROL.Objective):
    '''Subclass of ROL.Objective to define value and gradient for problem'''
    def __init__(self, inner_product):
        ROL.Objective.__init__(self)
        self.inner_product = inner_product
        self.m = Function(A, name="Controle_Continuo")
        self.t = Function(T)

    def value(self, x, tol):
        J = functional(self.m, self.t)
        return assemble(J)

    def gradient(self, g, x, tol):
        m = self.m
        t = self.t
        lam = solve_adjoint(m, t, functional(m, t))
        dmo = TestFunction(A)
        L = -2. * k(m) * kdash(m) * dmo * dx - kdash(m) * dmo * inner(grad(t), grad(lam) ) * dx #inner(k(m)*grad(t), grad(v))*dx
        deriv = assemble(L)
        if self.inner_product is not None:
            grad_ = self.inner_product.riesz_map(deriv)
        else:
            grad_ = deriv
        g.scale(0)
        g.vec += grad_
        self.deriva = grad_

    def update(self, x, flag, iteration):
        m = Function(A, x.vec, name="Controle_Continuo")
        self.m.assign(m)
        t = resp_direto(self.m)
        self.t.assign(t)
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
