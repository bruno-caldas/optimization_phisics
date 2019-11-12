## Sensitivity Analysis of Stokes flow
# Considerations
The q value was fixed at q=0.01 in order to simplify the problem and that is the reason the result **control.pvd** does not get in the global optimum.\
This code uses Dolfin-Adjoint for the sake of simplicity, however it can also be done purely with Fenics.

# Simulation
All the code used in this simulation can be found in this Github repository [Sensitivity_Analysis_Stokes.py](Sensitivity_Analysis_Stokes.py) .

# Results
The sensitivity can be found for each iteration in **output/derivative.pvd**.
Here's our logo (hover to see the title text):

Final Derivatie Plot:
![Plot of Derivative](output/derivative.png "Derivative")


Final Design Variable:
![Plot of Design Variable](output/control.png "Design Variable")


\ \
The results of the control variable can be found in **output/control.pvd** and the sensitivity in **output/derivatives_999.vtu** where 999 is the iteration wanted.\
The output file look like:
```
<UnstructuredGrid>
<Piece  NumberOfPoints="2601" NumberOfCells="5000">
<Points>
<DataArray  type="Float64"  NumberOfComponents="3"  format="ascii">
0 0 0  0.03 0 0  0.06 0 0  0.09 0 0  0.12 0 0  0.15 0 0  0.18 0 0
0.21 0 0  0.24 0 0  0.27 0 0  0.3 0 0  0.33 0 0  0.36 0 0  0.39 0 0
0.42 0 0  0.45 0 0  0.48 0 0  0.51 0 0  0.54 0 0  0.57 0 0  0.6 0 0
0.63 0 0  0.66 0 0  0.6899999999999999 0 0  0.72 0 0  0.75 0 0  0.78
0 0  0.8100000000000001 0 0  0.84 0 0  0.87 0 0  0.9 0 0  0.93 0 0
0.96 0 0  0.99 0 0  1.02 0 0  1.05 0 0  1.08 0 0  1.11 0 0  1.14 0 0
1.17 0 0  1.2 0 0  1.23 0 0  1.26 0 0  1.29 0 0  1.32 0 0  1.35 0 0
....
</PointData>
</Piece>
</UnstructuredGrid>
</VTKFile>
```
