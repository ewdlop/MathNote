import numpy as np
import sympy as sp

class GeneralizedStokesTheorem:
    """
    Comprehensive implementation of Generalized Stokes' Theorem
    
    Stokes' Theorem relates a surface integral of curl F to a line integral of F around the boundary of the surface.
    
    Mathematical Form: ∫∫S (curl F) · dS = ∮∂S F · dr
    
    Key Components:
    1. Symbolic curl calculation
    2. Surface parameterization
    3. Boundary curve integration
    4. Verification of the theorem
    """
    
    @staticmethod
    def symbolic_curl(vector_field):
        """
        Calculate symbolic curl of a vector field
        
        Parameters:
        vector_field (list): Symbolic vector field components [F1, F2, F3]
        
        Returns:
        list: Symbolic curl components [curl_x, curl_y, curl_z]
        """
        x, y, z = sp.symbols('x y z')
        
        # Symbolic partial derivatives
        curl_x = sp.diff(vector_field[2], y) - sp.diff(vector_field[1], z)
        curl_y = sp.diff(vector_field[0], z) - sp.diff(vector_field[2], x)
        curl_z = sp.diff(vector_field[1], x) - sp.diff(vector_field[0], y)
        
        return [curl_x, curl_y, curl_z]
    
    @staticmethod
    def surface_integral(vector_field, surface_parameterization):
        """
        Calculate surface integral of curl F
        
        Parameters:
        vector_field (list): Symbolic vector field components
        surface_parameterization (callable): Surface parameterization function
        
        Returns:
        Symbolic surface integral
        """
        # Symbolic parameters for surface
        u, v = sp.symbols('u v')
        
        # Surface normal calculation
        r_u, r_v = sp.symbols('r_u r_v', cls=sp.Function)
        
        # Compute surface normal via cross product of tangent vectors
        surface_normal = sp.Matrix([
            r_u(u,v)[1] * r_v(u,v)[2] - r_u(u,v)[2] * r_v(u,v)[1],
            r_u(u,v)[2] * r_v(u,v)[0] - r_u(u,v)[0] * r_v(u,v)[2],
            r_u(u,v)[0] * r_v(u,v)[1] - r_u(u,v)[1] * r_v(u,v)[0]
        ])
        
        # Symbolic curl computation
        curl_components = sp.Matrix(GeneralizedStokesTheorem.symbolic_curl(vector_field))
        
        # Surface integral computation
        surface_int = sp.integrate(
            sp.simplify(curl_components.dot(surface_normal)),
            (u, 0, 1),  # Example integration limits
            (v, 0, 1)   # Can be modified based on specific surface
        )
        
        return surface_int
    
    @staticmethod
    def line_integral(vector_field, boundary_curve):
        """
        Calculate line integral around surface boundary
        
        Parameters:
        vector_field (list): Symbolic vector field components
        boundary_curve (callable): Boundary curve parameterization
        
        Returns:
        Symbolic line integral
        """
        # Symbolic parameter for curve
        t = sp.Symbol('t')
        
        # Curve derivative (tangent vector)
        curve_derivative = sp.diff(boundary_curve(t), t)
        
        # Convert vector field to symbolic vector
        F = sp.Matrix(vector_field)
        
        # Line integral computation
        line_int = sp.integrate(
            sp.simplify(F.dot(curve_derivative)),
            (t, 0, 1)  # Example integration limits
        )
        
        return line_int
    
    @staticmethod
    def verify_stokes_theorem(vector_field, surface_param, boundary_curve):
        """
        Verify Stokes' Theorem by comparing surface and line integrals
        
        Parameters:
        vector_field (list): Symbolic vector field components
        surface_param (callable): Surface parameterization
        boundary_curve (callable): Boundary curve parameterization
        
        Returns:
        Comparison of surface and line integral results
        """
        # Compute surface integral of curl
        surface_integral_result = GeneralizedStokesTheorem.surface_integral(
            vector_field, surface_param
        )
        
        # Compute line integral along boundary
        line_integral_result = GeneralizedStokesTheorem.line_integral(
            vector_field, boundary_curve
        )
        
        return {
            'Surface Integral (curl F · dS)': surface_integral_result,
            'Line Integral (F · dr)': line_integral_result,
            'Theorem Verification': sp.simplify(
                surface_integral_result - line_integral_result
            )
        }

def main():
    # Symbolic setup
    x, y, z = sp.symbols('x y z')
    
    # Example vector field
    vector_field = [y, z, x]
    
    # Example surface parameterization (simple plane)
    def surface_param(u, v):
        return sp.Matrix([u, v, u*v])
    
    # Example boundary curve (square boundary)
    def boundary_curve(t):
        return sp.Matrix([
            t,          # x coordinate
            t,          # y coordinate
            t*t         # z coordinate
        ])
    
    # Compute curl
    curl_F = GeneralizedStokesTheorem.symbolic_curl(vector_field)
    print("Curl of Vector Field:", curl_F)
    
    # Verify Stokes' Theorem
    verification = GeneralizedStokesTheorem.verify_stokes_theorem(
        vector_field, 
        surface_param, 
        boundary_curve
    )
    
    print("\nStokes' Theorem Verification:")
    for key, value in verification.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
