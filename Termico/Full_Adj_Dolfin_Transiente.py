#-*- coding: utf-8 -*-
# Otimizacao de Sistema Termico Transiente com Sensibilidades Calculadas pelo Dolfin Adjoint
# Script criado por:
#       Diego Silva Prado           prado.dis@gmail.com
#       Bruno Caldas de Souza

from dolfin import *
from dolfin_adjoint import *

set_log_level(30)
CRED = '\033[91m'
CYELLOW = '\33[33m'
CEND = '\033[0m'
CGREEN  = '\33[32m'
CBEIGE  = '\33[36m'
CBOLD = '\33[1m'
CVIOLET = '\33[35m'

print(" ╔════════════════════════════════════════════════════════════════════════════════════════════╗")
print(" ║"+ CGREEN + CBOLD + " Otimizacao de Sistema Termico Transiente com Sensibilidades Calculadas pelo Dolfin Adjoint "+CEND+"║")
print(" ║"+ CGREEN + " Script criado por Diego Silva Prado e Bruno Caldas de Souza                                "+CEND+"║")
print(" ╚════════════════════════════════════════════════════════════════════════════════════════════╝")

pasta = "dolfin_transiente_full/"

#Propriedades do material
rho = Constant(1.)
cp = Constant(1.)

#Parametros do modelo de material
eps = Constant(1.0e-8)
p = Constant(5.)
alpha = Constant(1.0e-8)
beta = Constant(1.0e0)

#Transiente
dt = 0.2
tfin = 5.

#Condicoes de Contorno
Tpoint = 3000. #Temperatura no centro
Tamb = 273. #Temperatura ambiente

#Resolucao da malha
N = 120

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
    #bc2 = DirichletBC(T, 1000., "(x[0]-0.5)*(x[0]-0.5)+(x[1]-0.5)*(x[1]-0.5)<0.05*0.05")
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
    file_t = File(pasta + nome + ".pvd")
    while tfin+.0000000001 > ts:
        (F, bc1, ) = prob_direto(m, t, t0)
        stststr = Stsreturn(ts)
        print(' ║'+CRED+stststr+CEND+'║  Time: %.1f s / '+ str(tfin) + 's\r'.format(ts)) % ts,
        solve(F==0, t, [bc1])
        Jtemp = assemble((t*t+beta*((m**2)*(1-m)**2))*dx)
        try:
            Jlist
        except NameError:
            Jlist = [Jtemp]
        else:
            Jlist.append(Jtemp)
        file_t << t
        t0.assign(t)
        ts = ts+dt
    print ''
    return t, Jlist

# Arquivo para gravar a evolucao da otimizacao
file_m = File(pasta + "Var_Controle.pvd")
m_opt = Function(A, name = "controle")
def eval_cb(j, rho):
    m_opt.assign(rho)
    file_m << m_opt

#Codigo principal
#Criando variavel: Temperatura
t = Function(T)
t, Jlist = resp_direto(m,"problema_direto_inicial")

#Limites e restricao de volume
lb = 0.0
ub = 1.0
volume_constraint = UFLEqualityConstraint((0.3 - m)*dx, Control(m))

# Montando Funcional
J = 0.
for i in range(1, len(Jlist)):
    J += 0.5*(Jlist[i-1]+Jlist[i])*float(dt)
Jhat = ReducedFunctional(J, Control(m), eval_cb_post=eval_cb)

#Parametros do ROL solver
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

#Executando a otimizacao
print(CGREEN + " ╔════════════════════════════════════════════════════════════════════════════════════════════╗" + CEND)
print(CGREEN + " ║       Executando a Otimizacao com Sensibilidades Calculadas pelo Dolfin Adjoint            ║" + CEND)
print(CGREEN + " ╚════════════════════════════════════════════════════════════════════════════════════════════╝" + CEND)
solver = ROLSolver(problem, params, inner_product="L2")
m_opt = solver.solve()

#Gerandok resultados
file_m << m_opt

#Verificando a variavel depois de otimizado
print CBOLD + CVIOLET + 'Verificando Distribuicao Obtida no Problema Direto:' + CEND
t_pos = Function(T)
t_pos.rename("temperatura_opt","label")
t_pos = resp_direto(m_opt,"problema_direto_otimizado")

print CBEIGE + " ╔════════════════════════════════════════════════════════════════════════════════════════════╗"
print " ║              ♫                     Analise Concluida                         ♫             ║"
print " ╚════════════════════════════════════════════════════════════════════════════════════════════╝"+ CEND
