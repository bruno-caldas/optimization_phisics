from dolfin import *
from dolfin_adjoint import *

pasta = "dolfin_adjoint/"
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
    return (alphaunderbar - alphabar) * (1 * (1 + q) / (rho + q) - rho * (1 + q)/((rho + q)*(rho + q)))

mesh = RectangleMesh(mpi_comm_world(), Point(0.0, 0.0), Point(1.0, 1.0), N, N, 'crossed') # m
A = FunctionSpace(mesh, "DG", 0)        # control function space
T = FunctionSpace(mesh, "CG", 1)

t = Function(T)
v = TestFunction(T)

m = interpolate(Constant(0.5), A)

bc = [DirichletBC(T, 273., "on_boundary"), DirichletBC(T, 1000., "(x[0]-0.5)*(x[0]-0.5)+(x[1]-0.5)*(x[1]-0.5)<0.1*0.1")]
F = inner(k(m)*grad(t), grad(v))*dx

solve(F==0, t, bc)
File(pasta + "Resposta_Direto.pvd") << t

m_file = File(pasta + "Variavel_projeto.pvd")

lb = 0.0
ub = 1.0
#volume_constraint = UFLInequalityConstraint((0.3 - m)*dx, Control(m))
volume_constraint = UFLEqualityConstraint((0.3 - m)*dx, Control(m))
J = assemble(t*dx -inner((k(m)), (k(m)) )*dx )
Jhat = ReducedFunctional(J, Control(m), )

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
            'Iteration Limit': 7}
        }

solver = ROLSolver(problem, params, inner_product="L2")
m_opt = solver.solve()
m_opt.rename("controle", "label")
m_file << m_opt

t_pos = Function(T)
F = inner(k(m_opt)*grad(t_pos), grad(v))*dx
solve(F==0, t_pos, bc)
File(pasta + "pos.pvd") << t_pos
