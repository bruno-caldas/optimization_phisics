# -*- coding: utf-8 -*-
"""
Structural Classic Problem for Topology Optimization Method
with the help of Fenics and Pyipopt
Code written by Bruno Caldas 'bruno.caldas@gmail.com'
"""
from dolfin import *
import numpy
import pyipopt

#First let's define where the results should be saved
pasta = "continuo_adjoint_ipopt/"
m_file = File(pasta + "Variavel_projeto_continuo.pvd")
control_file = File(pasta + "control.pvd")

#The properties are bellow defined as global scope
Eo = Constant(10.)          #E0
nu = Constant(0.3)          #Nu
p = Constant(1.0)           #Penal Factor
eps = 1.e-7                 #Epsilon
carga = Constant((0.,-1.))  #Force applied (direction and intensity)
volcstr = 0.5               #Volume Constraint to be applied Vol<=V
r_min = 0.05                #Radius of Filtering
lb = 0.0                    #Lower limit of each control variable
ub = 1.0                    #Higher limit of each control variable

#The mesh and Function Spaces are defined bellow
N = 50                      #Degree of elements to be discritize
mesh = RectangleMesh(Point(0.0, 0.0),\
                    Point(2.0, 1.0),\
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
    """
    Material Model for SIMP. 'm' is the control variable and p the penalty
    factor. If you change here, you should change the xdash function as well.
    """
    materialModel = (eps + (1 - eps) * m**p)
    return materialModel
def xdash(m):
    """
    The derivative of the material model chosen in 'x(m)' python function.
    """
    dot_materialModel = ( p * (1 - eps) * m**(p-1))
    return dot_materialModel

def sigma(u_):
    """
    The hook's law in order to calculate the stress. The u_ in this case
    is the deformation of the structure
    """
    mu = Eo/(2.0*(1.0+nu))               # Lame parameter
    lm = Eo*nu/((1.0+nu)*(1.0-2.0*nu))   # Lame parameter
    I = Identity(2)
    stress = 2*mu*sym(grad(u_)) + lm*tr(sym(grad(u_)))*I
    return stress

def functional(m, u):
    """
    The basic Functional to minimize. Notice that if you want to change here,
    you should check the solve_adjoint function
    """
    Functional = inner(carga,u)*ds(1)
    return Functional

def density_filter(m, r_min):
    """
    The Lazarav Filter in order to implement deal with minimum length scale
    and checkboard
    """
    mt = Function(m.function_space() )
    w = TestFunction(m.function_space() )
    eq_l = (r_min**2)*inner(grad(mt), grad(w))*dx + mt*w*dx - m*w*dx
    solve(eq_l== 0, mt )
    File(pasta + "densidade.pvd") << mt
    return mt

def direct_problem(m, u):
    """
    The direct problem that needs to be satisfied during optimization.
    It returns the equation and boundary conditions both as a tupple
    """
    vtst = TestFunction(u.function_space())
    F = inner(x(m)*sigma(u),grad(vtst))*dx - inner(carga,vtst)*ds(1)
    bc = DirichletBC(U, (0.,0.), sub_domains, 2)
    return (F, bc)

def solution_direct_problem(m):
    """
    The solution of direct problem described by 'direct_problem' function.
    It returns the results, i.e the deformation of the structure
    """
    u = Function(U, name="strain")
    (F, bc) = direct_problem(m, u)
    solve(F==0, u, bc)
    return u

def solve_adjoint(m, u, J):
    """
    Function that gives the adjoint solution in order to compute later
    the gradient for the optimization
    """
    (F, bc) = direct_problem(m, u)
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
    df2dmt = r_min**2*inner(grad(adj2),grad(adj2_tst))*dx + \
                inner(adj2, adj2_tst)*dx - \
                r_min**2*inner(n,grad(adj2))*adj2_tst*ds
    df1dmt = xdash(mt)*inner(sigma(u), grad(adj))*adj2_tst*dx
    solve(df2dmt + df1dmt ==0, adj2)
    return adj2

def OptSolver(bounds, OptObj, max_iter=10):
    """
    Function that returns a pyipopt object with OptObj atributes
    """

    # Number of Design Variables
    nvar = OptObj.nvars
    # Upper and lower bounds
    x_L = numpy.ones(nvar) * bounds[0]
    x_U = numpy.ones(nvar) * bounds[1]
    # Number of non-zeros gradients
    constraints_nnz = nvar*OptObj.cst_num
    acst_L = numpy.array(OptObj.cst_L)
    acst_U = numpy.array(OptObj.cst_U)

    PyIpOptObj = pyipopt.create(nvar,               # number of the design variables
                         x_L,                       # lower bounds of the design variables
                         x_U,                       # upper bounds of the design variables
                         OptObj.cst_num,            # number of constraints
                         acst_L,                    # lower bounds on constraints,
                         acst_U,                    # upper bounds on constraints,
                         constraints_nnz,           # number of nonzeros in the constraint Jacobian
                         0,                         # number of nonzeros in the Hessian
                         OptObj.obj_fun,            # objective function
                         OptObj.obj_dfun,           # gradient of the objective function
                         OptObj.cst_fval,           # constraint function
                         OptObj.jacobian )          # gradient of the constraint function

    #Parameters
    PyIpOptObj.num_option('acceptable_tol', 1.0e-10)
    PyIpOptObj.num_option('eta_phi', 1e-12)                 # eta_phi: Relaxation factor in the Armijo condition.
    PyIpOptObj.num_option('theta_max_fact', 30000)	        # Determines upper bound for constraint violation in the filter.
    PyIpOptObj.int_option('max_soc', 20)
    PyIpOptObj.int_option('max_iter', max_iter)
    PyIpOptObj.int_option('watchdog_shortened_iter_trigger', 20)
    PyIpOptObj.int_option('accept_after_max_steps', 5)
    pyipopt.set_loglevel(1)                                 # turn off annoying pyipopt logging
    PyIpOptObj.int_option('print_level', 6)                 # very useful IPOPT output

    return PyIpOptObj

class IpoptClass:
    """
    This Class supplies a python object that has all Ipopt needs,
    such as ObjFun, Gradientes and Constraints
    """
    def __init__(self, rho_var):
        self.objfun_rf = None
        self.iter_fobj = 0
        self.iter_dobj = 0
        self.cst_U = []
        self.cst_L = []
        self.cst_num = 0
        self.rho_var = Function( rho_var.function_space() )
        self.nvars = len(self.rho_var.vector())
        self.state = solution_direct_problem(self.rho_var)

    def __check_ds_vars__(self, xi):
        """
        Method which checks the design variables
        """
        chk_var = False
        try: #If self.xi_array has yet not been defined
            xi_eval = self.xi_array - xi
            xi_nrm  = numpy.linalg.norm(xi_eval)
            if xi_nrm > 1e-16:
                self.xi_array = numpy.copy(xi)
                chk_var = True
        except AttributeError as error:
            self.xi_array = numpy.copy(xi)
            chk_var = True

        if chk_var is True:
            self.rho_var.vector()[:] = xi
        else:
            print(" *** Recycling the design variables...")
        ds_vars = self.rho_var
        return ds_vars

    def __vf_fun_var_assem__(self):
        rho_var_tst   = TestFunction(self.rho_var.function_space())
        self.vol_xi  = assemble(rho_var_tst * Constant(1) * dx)
        self.vol_sum = self.vol_xi.sum()

    def add_plot_res(self, file_out):
        self.file_out = file_out

    def obj_fun(self, rho_var, user_data=None):
        print(" \n **********************************" )
        print(" Objective Function Evaluation" )
        ds_vars = self.__check_ds_vars__(rho_var)
        fval = assemble( functional(ds_vars, self.state) )
        print(" fval: {}" .format(fval) )
        print(" ********************************** \n " )
        if self.file_out is not None:
            self.rho_var.rename("control", "label")
            self.file_out << self.rho_var
        self.iter_fobj += 1
        return fval

    def obj_dfun(self, xi, user_data=None):
        print(" \n **********************************" )
        print(" \n Objective Function Gradient Evaluation \n" )
        ds_vars  = self.__check_ds_vars__(xi)
        lam = solve_adjoint(ds_vars, self.state, functional(ds_vars, self.state))
        dmo = TestFunction( self.rho_var.function_space() )
        L = lam *dmo * dx
        deriv = assemble(L)
        grad_ = deriv
        dfval = deriv
        self.iter_dobj += 1
        return numpy.array( dfval)

    def add_volf_constraint(self, upp, lwr):
        self.__vf_fun_var_assem__()
        self.cst_U.append(upp)
        self.cst_L.append(lwr)
        self.cst_num += 1

    def volfrac_fun(self, xi):
        self.__check_ds_vars__(xi)
        volume_val = float( self.vol_xi.inner( self.rho_var.vector() ) )
        return volume_val/self.vol_sum

    def volfrac_dfun(self, xi=None, user_data=None):
        return numpy.array(self.vol_xi)/self.vol_sum

    def flag_jacobian(self):
        rows = []
        for i in range(self.cst_num):
            rows += [i] * self.nvars
        cols = list(range(self.nvars)) * self.cst_num
        return (numpy.array(rows, dtype=numpy.int), numpy.array(cols, dtype=numpy.int))

    def cst_fval(self, xi, user_data=None):
        cst_val = numpy.array(self.volfrac_fun(xi), dtype=numpy.float)
        return cst_val.T

    def jacobian(self, xi, flag=False, user_data=None):
        print(" \n Constraint Gradient Evaluation \n" )
        if flag:
            dfval = self.flag_jacobian()
        else:
            print( "CST Value:", self.volfrac_fun(xi) )
            dfval = self.volfrac_dfun()
        return dfval

if __name__ == '__main__':
    """
    This let the file to be executed only if asked directly, i.e.
    avoiding to any 'import .' outside here would work.
    """
    m = interpolate(Constant(0.5), A)

    penalty_numbers = [2, 3, 5, 7]
    for penal in penalty_numbers:
        p.assign(float(penal))
        fval = IpoptClass(m)
        fval.add_plot_res(control_file)
        fval.add_volf_constraint(volcstr, 0)
        lb = 0.0
        ub = 1.0
        bounds = [lb, ub]
        nlp = OptSolver(bounds, fval, 15)
        x0 = numpy.copy(m.vector())
        m.vector()[:], zl, zu, constraint_multipliers, obj, status = nlp.solve(x0)
        del fval, nlp
