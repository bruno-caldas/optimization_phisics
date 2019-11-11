# -*- coding: utf-8 -*-
"""
Structural Classic Problem for Topology Optimization Method
with the help of Fenics and ROL (BFGS)
Code written by Bruno Caldas 'bruno.caldas@gmail.com'
"""
from dolfin import *
from ROL.dolfin_vector import DolfinVector as FeVector
import numpy
import ROL
import copy

#First let's define where the results should be saved
pasta = "continuo_adjoint_rol/"
control_file = File(pasta + "control.pvd")

#The properties are bellow defined as global scope
Eo = Constant(10.)          #E0
nu = Constant(0.3)          #Nu
p = Constant(1.0)           #Penal Factor
eps = 1.e-7                 #Epsilon
carga = Constant((0.,-1.))  #Force applied (direction and intensity)
volcstr = 0.5               #Volume Constraint to be applied Vol<=V
r_min = 0                #Radius of Filtering
lb = 0.0                    #Lower limit of each control variable
ub = 1.0                    #Higher limit of each control variable

#The mesh and Function Spaces are defined bellow
N = 50                      #Degree of elements to be discritize
delta = 2.0
mesh = RectangleMesh(Point(0.0, 0.0),\
                    Point(delta, 1.0),\
                    2*N, N,\
                    'crossed')                  # Mesh Defined to be used
A = FunctionSpace(mesh, "CG", 1)                # Control Function Space
U = VectorFunctionSpace(mesh, "CG", 1, dim=2)   # Solution Function Space
u = Function(U, name="strain")                  #Strain Vector Value from U
v = TestFunction(U)                             # T.F. for variational problem

# The place to apply the force is described as subdomain as bellow
sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
sub_domains.set_all(0)
CompiledSubDomain("on_boundary && x[0]>=1.9 && x[1]<=0.01" ).mark(sub_domains, 1)
CompiledSubDomain("on_boundary && x[0]<=0.01" ).mark(sub_domains, 2)
File(pasta +"subdominios.pvd") << sub_domains
ds = Measure('ds', domain = mesh, subdomain_data = sub_domains)

#Definition of all python functions necessary for the problem

def x(m):
    return (eps + (1 - eps) * m**p)
def xdash(m):
    return ( p * (1 - eps) * m**(p-1))

def sigma(u_):
    mu = Eo/(2.0*(1.0+nu))               # Lame parameter
    lm = Eo*nu/((1.0+nu)*(1.0-2.0*nu))   # Lame parameter
    I = Identity(2)
    return 2*mu*sym(grad(u_)) + lm*tr(sym(grad(u_)))*I

def density_filter(m, r_min):
    if int(r_min) == 0:
        return m
    else:
        mt = Function(m.function_space() )
        w = TestFunction(m.function_space() )
        eq_l = (r_min**2)*inner(grad(mt), grad(w))*dx + mt*w*dx - m*w*dx
        solve(eq_l== 0, mt )
        return mt

def prob_direto(m, u):
    vtst = TestFunction(u.function_space())
    F = inner(x(m)*sigma(u),grad(vtst))*dx - inner(carga,vtst)*ds(1)
    bc = DirichletBC(U, (0.,0.), sub_domains, 2)
    return (F, bc)

def resp_direto(m):
    u = Function(U, name="strain")
    (F, bc) = prob_direto(m, u)
    solve(F==0, u, bc)
    return u
def solve_adjoint(m, u, J):
    """
    Function that gives the adjoint solution in order to compute later
    the gradient for the optimization
    """
    (F, bc) = prob_direto(m, u)
    bc.homogenize() #adjoint has homogeneous BCs. It means(DirichletBC(U, Constant(0), 'on_boundary')
    adj = Function(U)
    adj_tst = TestFunction(U)
    mt = density_filter(m, r_min)
    #Let's start with the main terms of the adjoint equation
    adjEquation = x(mt)*inner(sigma(adj), grad(adj_tst))*dx - \
                    inner(carga, adj_tst)*ds(1)
    solve(adjEquation==0, adj, bc)
    #Now is the point to implement the filtering terms
    adj2 = Function(A)
    adj2_tst = TestFunction(A)
    n = FacetNormal(mesh)
    if int(r_min) == 0:
        df2dmt = inner(adj2, adj2_tst)*dx
    else:
        df2dmt = r_min**2*inner(grad(adj2),grad(adj2_tst))*dx + \
                    inner(adj2, adj2_tst)*dx - \
                    r_min**2*inner(n,grad(adj2))*adj2_tst*ds
    df1dmt = xdash(mt)*inner(sigma(u), grad(adj))*adj2_tst*dx
    solve(df2dmt + df1dmt ==0, adj2)
    return adj2

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

m_file = File(pasta + "Variavel_projeto_continuo.pvd")
lb = 0.0
ub = 1.0
def functional(m, u):
    return inner(carga,u)*ds(1)

class ObjR(ROL.Objective):
    '''Subclass of ROL.Objective to define value and gradient for problem'''
    def __init__(self, inner_product):
        ROL.Objective.__init__(self)
        self.inner_product = inner_product
        self.m = Function(A, name="Controle_Continuo")
        self.u = Function(U)

    def value(self, xm, tol):
        J = functional(self.m, self.u)
        return assemble(J)

    def gradient(self, g, xm, tol):
        m = self.m
        u = self.u
        lam = solve_adjoint(m, u, functional(m, u))
        dmo = TestFunction(A)
        #L = -xdash(m) * dmo * inner(sigma(u), grad(lam) ) * dx
        L = lam *dmo * dx
        deriv = assemble(L)
        if self.inner_product is not None:
            grad_ = self.inner_product.riesz_map(deriv)
        else:
            grad_ = deriv
        g.scale(0)
        g.vec += grad_
        self.deriva = grad_

    def update(self, xm, flag, iteration):
        m = Function(A, xm.vec, name="Controle_Continuo")
        self.m.assign(m)
        u = resp_direto(self.m)
        self.u.assign(u)
        if iteration >= 0:
            control_file << self.m

    def resposta(self):
        return self.m


class VolConstraint(ROL.Constraint):
    def __init__(self, inner_product):
        ROL.Constraint.__init__(self)
        self.inner_product = inner_product
    def value(self, cvec, xvec, tol):
        a = Function(A, xvec.vec)
        val = assemble(a * dx) - volcstr * delta
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

m_n = interpolate(Constant(0.5), A)
m = density_filter(m_n, r_min)
m = FeVector(m.vector(), dot_product)

lower = interpolate(Constant(0.0), A)
lower = FeVector(lower.vector(), dot_product)
upper = interpolate(Constant(1.0), A)
upper = FeVector(upper.vector(), dot_product)

obj = ObjR(dot_product)
volConstr = VolConstraint(dot_product)

set_log_level(30)

params = {
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
            'Iteration Limit': 1}
        }

params_i = ROL.ParameterList(params, "Parameters")
bound_constraint = ROL.Bounds(lower, upper, 1.0)
optimProblem = ROL.OptimizationProblem(obj, m, bnd=bound_constraint, econ=volConstr, emul=l)
solver = ROL.OptimizationSolver(optimProblem, params_i)

solver.solve()

params_f =copy.deepcopy(params)
params_f['Step']['Augmented Lagrangian']['Subproblem Iteration Limit'] = 30
params_f = ROL.ParameterList(params_f, "Parameters")

penalizadores = [2, 3, 5]
for penal in penalizadores:
    p.assign(float(penal))
    m = obj.resposta()
    m = FeVector(m.vector(), dot_product)
    optimProblem = ROL.OptimizationProblem(obj, m, bnd=bound_constraint, econ=volConstr, emul=l)
    del solver
    solver = ROL.OptimizationSolver(optimProblem, params_f)
    try:
        solver.solve()
    except:
        pass
