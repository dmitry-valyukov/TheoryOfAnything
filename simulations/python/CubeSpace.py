# Let's perform a symbolic evaluation to see the force scaling of an elastic network where bonds have an initial tension.
# Consider a displacement 'x' of a node in a 3D network, and let's check the restoring force scaling.
# Alternatively, let's verify if a 3D integral of pairwise interactions scaling as 1/r gives a cubic stiffness response.

import sympy as sp
print("Sympy загружен из:", sp.__file__) # Этот принт покажет, какой именно файл перехватывает импорт

# Let two nodes be separated by distance r in 3D. Suppose there is a tension T(r) = c / r.
# If we displace a central node by x, the change in distance and the resulting restoring force components can be integrated.
# Let's set up a simple geometric configuration to see the force component.
x, y, z, c = sp.symbols('x y z c', real=True)
# Distance from displaced node (at x, 0, 0) to a point (u, v, w) in space
u, v, w = sp.symbols('u v w', real=True)
r_orig = sp.sqrt(u**2 + v**2 + w**2)
r_disp = sp.sqrt((u-x)**2 + v**2 + w**2)

# If the tension force along the line is F = c / r
# The force vector from (u,v,w) on the node at x is (c / r_disp) * (vector_disp / r_disp)
# Let's expand the x-component of the force for small x to find the leading order terms.
force_x_elem = (c / r_disp) * (u - x) / r_disp
# Expand in Taylor series around x = 0
force_expanded = sp.series(force_x_elem, x, 0, 5)
print(force_expanded)
