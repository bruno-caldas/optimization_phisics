from dolfin import *
from ROL.dolfin_vector import DolfinVector as FeVector
import os, shutil, datetime
import numpy
import ROL

import argparse
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--r", type=float, default=0.1)
parser.add_argument("--v", type=float, default=0.8)
args, _=parser.parse_known_args()
#python -i Trans_Vorticidade_Continuo_Navier_Stokes_Helmholtz.py --r 0.05 --v 0.8
#python -i Trans_Vorticidade_Continuo_Navier_Stokes_Helmholtz.py --r 0.07 --v 0.8
#python -i Trans_Vorticidade_Continuo_Navier_Stokes_Helmholtz.py --r 0.09 --v 0.8
#python -i Trans_Vorticidade_Continuo_Navier_Stokes_Helmholtz.py --r 0.1 --v 0.8
#python -i Trans_Vorticidade_Continuo_Navier_Stokes_Helmholtz.py --r 0.1 --v 0.7
#python -i Trans_Vorticidade_Continuo_Navier_Stokes_Helmholtz.py --r 0.1 --v 0.6

pasta = "TNS_rmin"+str(args.r)+"_v_"+str(args.v)+"/"
def erase_old_results(output_dir):
    current_dir = os.getcwd()
    #now = datetime.datetime.now()
    new_dir = current_dir + "/" + output_dir
    source_code_name = os.path.basename(__file__)
    #verify if exist
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        os.makedirs(new_dir +"/Source")
    if os.path.exists(new_dir + "/Source/"+source_code_name):
        os.remove(new_dir + "/Source/"+source_code_name)
        os.mknod(new_dir + "/Source/"+source_code_name)
    shutil.copy2(source_code_name,  new_dir + "/Source/" + source_code_name )
    if os.path.exists(new_dir + "/Source/Espec.txt"):
        os.remove(new_dir + "/Source/Espec.txt")
        os.mknod(new_dir + "/Source/Espec.txt")
    if os.path.exists(new_dir + "/Source/Objetivo.txt"):
        os.remove(new_dir + "/Source/Objetivo.txt")
        os.mknod(new_dir + "/Source/Objetivo.txt")
    os.chdir(new_dir)
    filelist = [ f for f in os.listdir(".") if f.endswith(".pvd") or f.endswith(".vtu") or f.endswith(".xdmf") ]
    for f in filelist:
        os.remove(f)
    return
erase_old_results(pasta)
mu = Constant(1.0)                   # viscosity
alphaunderbar = 2.5 * mu / (100**2)  # parameter for \alpha
alphabar = 2.5 * mu / (0.01**2)      # parameter for \alpha
q = Constant(0.01) # q value that controls difficulty/discrete-valuedness of solution

def alpha(rho):
    """Inverse permeability as a function of rho, equation (40)"""
    return alphabar + (alphaunderbar - alphabar) * rho * (1 + q) / (rho + q)

def alphadash(rho):
    return (alphaunderbar - alphabar) * (1 * (1 + q) / (rho + q) - rho * (1 + q)/((rho + q)*(rho + q)))

N = 40
delta = 1.5  # The aspect ratio of the domain, 1 high and \delta wide
#V = (1.0/2) * delta  # want the fluid to occupy 1/3 of the domain
V = (args.v) * delta  # want the fluid to occupy 1/3 of the domain
r_min = Constant(args.r) #era 0.1
dtempo = .01
Tfim = 1#2
veloc = File("veloc.pvd")
press = File("pressao.pvd")

mesh = RectangleMesh(mpi_comm_world(), Point(0.0, 0.0), Point(delta, 1.0), int(N*delta), N, 'crossed')
A = FunctionSpace(mesh, "CG", 1)        # control function space

U_h = VectorElement("CG", mesh.ufl_cell(), 2, dim=2)
P_h = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, U_h*P_h)          # mixed Taylor-Hood function space
U = VectorFunctionSpace(mesh, "CG", 2, dim=2)  # velocity function space

class InflowOutflow(Expression):
    def eval(self, values, x):
        values[1] = 0.0
        values[0] = 0.0
        l = 1.0/6.0
        gbar = 1000.0
        if x[0] == 0.0 or x[0] == delta:
            """if (1.0/4 - l/2) < x[1] < (1.0/4 + l/2):
                t = x[1] - 1.0/4
                values[0] = gbar*(1 - (2*t/l)**2)"""
            if (3.0/4 - l/2) < x[1] < (3.0/4 + l/2):
                t = x[1] - 3.0/4
                values[0] = gbar*(1 - (2*t/l)**2)
    def value_shape(self):
        return (2,)

def density_filter(m, r_min):
    mt = TrialFunction(m.function_space() )
    mtresp = Function(m.function_space() )
    w = TestFunction(m.function_space() )
    n = FacetNormal(mesh)
    eq_l = (r_min**2)*inner(grad(mt), grad(w))*dx + mt*w*dx - m*w*dx -r_min**2*inner(n,grad(mt))*w*ds
    solve(lhs(eq_l)== rhs(eq_l), mtresp )
    File("densidade.pvd") << mtresp
    return mtresp

def prob_direto(rho, w, u0):
    W = w.function_space()
    (u, p) = split(w)
    (v, q) = TestFunctions(W)
    f = Constant((0,0))
    rhot = density_filter(rho, r_min) #Filtro Helmholtz
    """F = ( \
        alpha(rhot) * inner(u, v) * dx + \
        mu * inner(grad(u), grad(v)) * dx + \
        inner(dot(u, nabla_grad(u)), v) * dx + \
        inner(grad(p), v) * dx + \
        inner(div(u), q) * dx )"""
    #Inicial = (inner(grad(u), grad(v)) - div(v)*p + q*div(u))*dx + Constant(0) * q * dx
    Inicial = (alpha(rhot) * inner(u, v) * dx + mu * inner(grad(u), grad(v)) * dx +
         inner(grad(p), v) * dx  + inner(div(u), q) * dx + Constant(0) * q * dx)
    LInicial = 0#inner(f, v)*dx

    NS = (inner(u,v) + dtempo*(.5*inner(grad(u)*u0,v) - .5*inner(grad(v)*u0,u)\
        + mu*inner(grad(u),grad(v)) - div(v)*p) + q*div(u) + alpha(rhot)*inner(u,v)\
        )*dx
    LNS = inner(u0,v)*dx
    bc = DirichletBC(W.sub(0), InflowOutflow(degree=1), "on_boundary")
    return (Inicial, NS, LNS, bc)

def resp_direto(rho):
    """Solve the forward problem for a given fluid distribution rho(x)."""
    wt = TrialFunction(W)
    w = Function(W)
    u0 = Function(U)
    (Inicial, NS, LNS, bc) = prob_direto(rho, wt, u0)
    solve(lhs(Inicial) == rhs(Inicial), w, bc)#,solver_parameters=dict(linear_solver="lu"))
    #(u, p) = split(w)
    (u, p) = w.split()
    u.rename("velocidade", "conforme_tempo")
    p.rename("pressao", "conforme_tempo")
    veloc << u
    press << p
    tempo = 0.
    while tempo <= Tfim:
        fa = FunctionAssigner(U, W.sub(0))
        fa.assign(u0,w.sub(0))
        (Inicial, NS, LNS, bc) = prob_direto(rho, wt, u0)
        solve(NS == LNS, w, bcs=[bc])#, solver_parameters=dict(linear_solver="lu"))
        veloc << u
        press << p
        tempo += float(dtempo)
    return w

def Funcional(rho, w):
    (u, p) = split(w)
    #return 0.5 * inner(alpha(rho) * u, u) * dx + mu * inner(grad(u), grad(u)) * dx
    return -inner(curl(u), curl(u))*dx + 1.e6*rho*(1.-rho)*dx

def adjunto(rho, w, J):
    u0 = Function(U)
    (Inicial, NS, LNS, bc) = prob_direto(rho, w, u0)
    bc.homogenize() # adjoint has homogeneous BCs
    adj = Function(w.function_space())
    (u_ad, p_ad) = split(adj)
    rhot = density_filter(rho, r_min) #Filtro Helmholtz

    adj_tst = TestFunction(W)
    (v_ad, q_ad) = split(adj_tst)
    n = FacetNormal(mesh)
    w_resp = resp_direto(rhot)
    u_resp, p_resp = split(w_resp)

    F_ad = ( mu * inner(grad(u_ad), grad(v_ad)) - \
        inner(grad(u_resp)*u_ad, v_ad) + \
        inner(grad(u_ad).T*u_resp, v_ad) + \
        alpha(rhot) * inner(u_ad, v_ad) - \
        inner(grad(p_ad), v_ad) + \
        q_ad*div(u_ad) ) * dx #- mu*inner(grad(v_ad)*n, u_ad)*ds

    """dJdu = (2 * mu * inner(grad(u_resp),grad(v_ad)) + #Derivada do Funcional
        alpha(rhot) * inner(u_resp, v_ad) )*dx"""
    dJdu = -2.*inner(curl(u_resp), curl(v_ad)) * dx

    print("Agora vai o adjunto !!!!!!!!!!!")
    solve(F_ad - dJdu ==0, adj, bc)

    adj_u, adj_p = split(adj)
    adj2 = Function(A)
    adj2_tst = TestFunction(A)
    n = FacetNormal(mesh)
    df3dmt = r_min**2*inner(grad(adj2),grad(adj2_tst))*dx + inner(adj2, adj2_tst)*dx -r_min**2*inner(n,grad(adj2))*adj2_tst*ds
    df1dmt = alphadash(rhot)*inner(u_resp, adj_u)*adj2_tst*dx
    solve(df3dmt + df1dmt ==0, adj2)

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

state_file = File("state.pvd")
control_file = File("control.pvd")
malha=XDMFFile(mpi_comm_world() , '/malha.xdmf')
class ObjR(ROL.Objective):
    '''Subclass of ROL.Objective to define value and gradient for problem'''
    def __init__(self, inner_product):
        ROL.Objective.__init__(self)
        self.inner_product = inner_product
        self.rho_o = Function(A, name="Control")
        self.state = Function(W)

    def value(self, x, tol):
        J = Funcional(self.rho_o, self.state)
        return assemble(J)

    def gradient(self, g, x, tol):
        rho_o = self.rho_o
        state = self.state
        (u, p) = split(state)
        lam = adjunto(rho_o, state, Funcional(rho_o, state))
        #(v, q)= split(lam)
        drho = TestFunction(A)
        #L = 0.5 * alphadash(rho_o) * drho * inner(u, u) * dx - alphadash(rho_o) * drho * inner(u, v) * dx
        L =  drho * lam * dx + 1.6*(1.-2.*rho_o) *drho*dx
        #L = - alphadash(rho_o) * drho * inner(u, lam) * dx #Sem Helmholtz
        deriv = assemble(L)
        if self.inner_product is not None:
            grad = self.inner_product.riesz_map(deriv)
        else:
            grad = deriv
        g.scale(0)
        g.vec += grad

    def update(self, x, flag, iteration):
        rho_o = Function(A, x.vec, name="Control")
        self.rho_o.assign(rho_o)
        state = resp_direto(self.rho_o)
        self.state.assign(state)
        #self.rho.rename("control", "label")
        if iteration >= 0:
            control_file << self.rho_o
            state_file << self.state
            #J = Funcional(self.rho_o, self.state)
            #self.funcio = assemble(J)
            #arquivo_objetivo
            (u, p) = split(self.state)
            self.objprincipal = assemble(-inner(curl(u), curl(u))*dx)
            self.regularizacao = assemble(1.6*self.rho_o*(1.-self.rho_o)*dx)
            espec=open(os.getcwd() +"/Source/Espec.txt","a")
            espec.write("Iteracao= \t %s \t" % iteration)
            espec.write("Obj= \t %s \t" % self.objprincipal)
            espec.write("rho(1-rho)= \t %s \n" % self.regularizacao)
            espec.close()
        self.iteracoes = iteration
        self.solparcial = rho_o

    def resposta(self):
        return self.rho_o

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
l_initializacao = ROL.StdVector(1)

x = interpolate(Constant(V/delta), A)
x = FeVector(x.vector(), dot_product)

lower = interpolate(Constant(0.0), A)
lower = FeVector(lower.vector(), dot_product)
upper = interpolate(Constant(1.0), A)
upper = FeVector(upper.vector(), dot_product)

# Instantiate Objective class for poisson problem
obj = ObjR(dot_product)
volConstr = VolConstraint(dot_product)

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
                    'Subproblem Iteration Limit'              : 5
                  }},
        'Status Test': {
            'Gradient Tolerance': 1e-15, 'Relative Gradient Tolerance': 1e-10,
            'Step Tolerance': 1e-16, 'Relative Step Tolerance': 1e-10,
            'Iteration Limit': 5}
        }
paramsDict2 = {
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
                    'Subproblem Iteration Limit'              : 3
                  }},
        'Status Test': {
            'Gradient Tolerance': 1e-15, 'Relative Gradient Tolerance': 1e-10,
            'Step Tolerance': 1e-16, 'Relative Step Tolerance': 1e-10,
            'Iteration Limit': 30}
        }
params = ROL.ParameterList(paramsDict, "Parameters")
bound_constraint = ROL.Bounds(lower, upper, 1.0)

optimProblem = ROL.OptimizationProblem(obj, x, bnd=bound_constraint, econ=volConstr, emul=l_initializacao)
solver = ROL.OptimizationSolver(optimProblem, params)
solver.solve()

print("aqui mudouuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu")

q.assign(0.1)
interm = obj.resposta()
del x
interm = FeVector(interm.vector(), dot_product)
params2 = ROL.ParameterList(paramsDict2, "Parameters")
bound_constraint = ROL.Bounds(lower, upper, 1.0)
optimProblem = ROL.OptimizationProblem(obj, interm, bnd=bound_constraint, econ=volConstr, emul=l_initializacao)
solver2 = ROL.OptimizationSolver(optimProblem, params2)
solver2.solve()
