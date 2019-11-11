from dolfin import *
from dolfin_adjoint import *
import copy

pasta = "dolfin_adjoint_rol/"
Eo = Constant(10.)
nu = Constant(0.3)
p = Constant(1.0)
eps = 1.e-7
b = Constant((0.,-1.))
N = 50
volcstr = 0.5

def x(m):
    return (eps + (1 - eps) * m**p)#*10000.

def sigma(u_):
    mu = Eo/(2.0*(1.0+nu))               # Lame parameter
    lm = Eo*nu/((1.0+nu)*(1.0-2.0*nu))   # Lame parameter
    I = Identity(2)
    return (2*mu*sym(grad(u_)) + lm*tr(sym(grad(u_)))*I)

mesh = RectangleMesh(Point(0.0, 0.0), Point(2.0, 1.0), 2*N, N, 'crossed') # m
A = FunctionSpace(mesh, "CG", 1)        # control function space
U = VectorFunctionSpace(mesh, "CG", 1, dim=2)
u = Function(U, name="strain")
v = TestFunction(U)

sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
sub_domains.set_all(0)
CompiledSubDomain("on_boundary && x[0]>=1.9 && x[1]<=0.01" ).mark(sub_domains, 1)
CompiledSubDomain("on_boundary && x[0]<=0.01" ).mark(sub_domains, 2)
File(pasta +"subdominios.pvd") << sub_domains
ds = Measure('ds', domain = mesh, subdomain_data = sub_domains)
m_n = interpolate(Constant(0.5), A)
r_min = 0.05
def density_filter(m_n, r_min):
    m_ = Function(m_n.function_space() )
    w = TestFunction(m_n.function_space() )
    eq_l = (r_min**2)*inner(grad(m_), grad(w))*dx + m_*w*dx - m_n*w*dx
    solve(eq_l== 0, m_ )
    return m_

m = density_filter(m_n, r_min)
bc = DirichletBC(U, (0.,0.), sub_domains, 2)
F = inner(x(m)*sigma(u),grad(v))*dx - inner(b,v)*ds(1)

solve(F==0, u, bc)
File(pasta+"resposta.pvd") << u

m_file = File(pasta + "Variavel_projeto.pvd")
m_viz = Function(A, name = "controle")
def eval_cb(j, rho):
    m_viz.assign(rho)
    m_file << m_viz
lb = 0.0
ub = 1.0

volume_constraint = UFLEqualityConstraint((volcstr - m)*dx, Control(m))
J = assemble(inner(b,u)*ds(1) )
Jhat = ReducedFunctional(J, Control(m), eval_cb_post=eval_cb)

problem = MinimizationProblem(Jhat, bounds=(lb, ub), constraints=volume_constraint)
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

solver = ROLSolver(problem, params, inner_product="L2")
try:
    m_opt = solver.solve()
except:
    pass

m_viz.rename("controle", "label")
m_file << m_viz

params_f =copy.deepcopy(params)
params_f['Step']['Augmented Lagrangian']['Subproblem Iteration Limit'] = 30

penalizadores = [2, 3, 5]
for cont in penalizadores:
    get_working_tape().clear_tape()
    p.assign(float(cont))
    m_n.assign(m_viz)
    m = density_filter(m_n, r_min)
    F = inner(x(m)*sigma(u),grad(v))*dx - inner(b,v)*ds(1)
    solve(F==0, u, bc)
    volume_constraint = UFLEqualityConstraint((volcstr - m)*dx, Control(m))
    J = assemble(inner(b,u)*ds(1) )
    Jhat = ReducedFunctional(J, Control(m), eval_cb_post=eval_cb)
    problem2 = MinimizationProblem(Jhat, bounds=(lb, ub), constraints=volume_constraint)
    if cont==penalizadores[-1]:
        solver = ROLSolver(problem2, params_f, inner_product="L2")
    else:
        solver = ROLSolver(problem2, params, inner_product="L2")
    try:
        m_opt = solver.solve()
    except:
        pass
