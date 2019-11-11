from dolfin import *
from ROL.dolfin_vector import DolfinVector as FeVector
import os, shutil, datetime
import numpy
import ROL
from mshr import *

res =80
pasta = "CFD_b_anlise_"+ str(res)

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
mu = Constant(1.0)

#mesh = RectangleMesh(mpi_comm_world(), Point(0.0, 0.0), Point(delta, 1.0), int(N*delta), N, 'crossed')
#Para ler
mesh = Mesh()
hdf = HDF5File(mesh.mpi_comm(), "../Resultado"+str(res)+".h5", "r")
hdf.read(mesh, "mesh", False)
A = FunctionSpace(mesh, "CG", 1)        # control function space
rho = Function(A)
hdf.read(rho, "solucao")

U_h = VectorElement("CG", mesh.ufl_cell(), 2, dim=2)
P_h = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, U_h*P_h)          # mixed Taylor-Hood function space
U = VectorFunctionSpace(mesh, "CG", 2, dim=2)  # velocity function space
veloc = File(pasta + "veloc.pvd")
press = File(pasta + "pressao.pvd")

alphaunderbar = 0.
alphabar = 2.5 * 1.e12
q = Constant(0.1) # q value that controls difficulty/discrete-valuedness of solution
dtempo = 0.1
delta = 1.5

def alpha(rho):
    """Inverse permeability as a function of rho, equation (40)"""
    return alphabar + (alphaunderbar - alphabar) * rho * (1 + q) / (rho + q)

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

def prob_direto(rho, w,u0):
    W = w.function_space()
    (u, p) = split(w)
    (v, q) = TestFunctions(W)
    f = Constant((0,0))
    #rhot = density_filter(rho, r_min) #Filtro Helmholtz
    """F = ( \
        alpha(rhot) * inner(u, v) * dx + \
        mu * inner(grad(u), grad(v)) * dx + \
        inner(dot(u, nabla_grad(u)), v) * dx + \
        inner(grad(p), v) * dx + \
        inner(div(u), q) * dx )"""
    #Inicial = (inner(grad(u), grad(v)) - div(v)*p + q*div(u))*dx + Constant(0) * q * dx
    Inicial = (alpha(rho) * inner(u, v) * dx + mu * inner(grad(u), grad(v)) * dx  \
        + inner(grad(p), v) * dx  + inner(div(u), q) * dx + Constant(0) * q * dx)
    LInicial = 0#inner(f, v)*dx

    NS = (inner(u,v) + dtempo*(.5*inner(grad(u)*u0,v) - .5*inner(grad(v)*u0,u)\
        + mu*inner(grad(u),grad(v)) - div(v)*p) + q*div(u) + alpha(rho)*inner(u,v)\
        )*dx
    LNS = inner(u0,v)*dx
    bc = DirichletBC(W.sub(0), InflowOutflow(degree=1), "on_boundary")
    return (Inicial, NS, LNS, bc)

def resp_direto(rho):
    """Solve the forward problem for a given fluid distribution rho(x)."""
    wt = TrialFunction(W)
    w = Function(W)
    u0 = Function(U)
    (Inicial, NS, LNS, bc) = prob_direto(rho, wt,u0)
    solve(lhs(Inicial) == rhs(Inicial), w, bc)#,solver_parameters=dict(linear_solver="lu"))
    #(u, p) = split(w)
    (u, p) = w.split()
    u.rename("velocidade", "conforme_tempo")
    p.rename("pressao", "conforme_tempo")
    veloc << u
    press << p
    tempo = 0.
    """while tempo <= Tfim:
        fa = FunctionAssigner(U, W.sub(0))
        fa.assign(u0,w.sub(0))
        (Inicial, NS, LNS, bc) = prob_direto(rho, wt, u0)
        solve(NS == LNS, w, bcs=[bc])#, solver_parameters=dict(linear_solver="lu"))
        veloc << u
        press << p
        tempo += float(dtempo)"""
    return w

def Funcional(rho, w):
    (u, p) = split(w)
    #return 0.5 * inner(alpha(rho) * u, u) * dx + mu * inner(grad(u), grad(u)) * dx
    return -inner(curl(u), curl(u))*dx + 1.e6*rho*(1.-rho)*dx

state = resp_direto(rho)
