import numpy as np
from typing import Callable, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy.integrate import quad, dblquad
from dataclasses import dataclass

import numpy as np
from typing import Callable, List, Tuple, Union, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import factorial

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Union
from scipy.integrate import quad, complex_ode
from scipy.special import factorial
import sympy as sp

class ComplexAnalysis:
    """Implementation of complex analysis theorems and concepts"""
    
    @staticmethod
    def cauchy_riemann_check(u: Callable, v: Callable, point: complex) -> bool:
        """
        Check if Cauchy-Riemann equations are satisfied at a point
        u_x = v_y and u_y = -v_x
        """
        h = 1e-7
        x, y = point.real, point.imag
        
        # Compute partial derivatives
        u_x = (u(x + h, y) - u(x, y)) / h
        u_y = (u(x, y + h) - u(x, y)) / h
        v_x = (v(x + h, y) - v(x, y)) / h
        v_y = (v(x, y + h) - v(x, y)) / h
        
        return np.isclose(u_x, v_y) and np.isclose(u_y, -v_x)

    @staticmethod
    def laurent_series(f: Callable, z0: complex, radius: float, n_terms: int) -> List[complex]:
        """
        Compute Laurent series coefficients around point z0
        Returns coefficients [a_-n, ..., a_-1, a_0, a_1, ..., a_n]
        """
        coefficients = []
        t = np.linspace(0, 2*np.pi, 1000)
        
        for n in range(-n_terms, n_terms + 1):
            def integrand(theta):
                z = z0 + radius * np.exp(1j * theta)
                return f(z) / (z - z0)**(n+1)
            
            real_part = quad(lambda x: np.real(integrand(x)), 0, 2*np.pi)[0]
            imag_part = quad(lambda x: np.imag(integrand(x)), 0, 2*np.pi)[0]
            
            coeff = (real_part + 1j*imag_part) / (2*np.pi*1j)
            coefficients.append(coeff)
            
        return coefficients

    @staticmethod
    def residue(f: Callable, pole: complex, order: int = 1) -> complex:
        """Calculate residue of function f at given pole"""
        if order == 1:
            h = 1e-7
            return (f(pole + h) * h).real + 1j * (f(pole + h) * h).imag
        else:
            factorial_term = factorial(order - 1)
            def derivative(z):
                return ((z - pole)**(order) * f(z))
            
            h = 1e-7
            return (derivative(pole + h) / factorial_term)

    @staticmethod
    def argument_principle(f: Callable, contour: List[complex]) -> Tuple[int, int]:
        """
        Apply argument principle to count zeros and poles
        Returns (number of zeros - number of poles) inside contour
        """
        def log_derivative(z):
            h = 1e-7
            return (np.log(f(z + h)) - np.log(f(z))) / h
        
        integral = 0
        for i in range(len(contour) - 1):
            z1, z2 = contour[i], contour[i + 1]
            integral += quad(lambda t: log_derivative(z1 + t*(z2-z1)), 0, 1)[0]
            
        return int(np.real(integral / (2*np.pi*1j)))

class FundamentalCalculus:
    """Implementation of Fundamental Theorems of Calculus"""
    
    @staticmethod
    def first_fundamental_theorem(f: Callable, a: float, b: float, n_points: int = 1000) -> float:
        """
        First Fundamental Theorem of Calculus
        Demonstrates that differentiation and integration are inverse operations
        """
        x = np.linspace(a, b, n_points)
        dx = x[1] - x[0]
        
        # Compute antiderivative
        F = np.cumsum(f(x)) * dx
        
        # Return F(b) - F(a)
        return F[-1] - F[0]
    
    @staticmethod
    def second_fundamental_theorem(F: Callable, f: Callable, x: float, h: float = 1e-7) -> bool:
        """
        Second Fundamental Theorem of Calculus
        Verifies that d/dx ∫(from a to x) f(t)dt = f(x)
        """
        derivative = (F(x + h) - F(x)) / h
        return np.isclose(derivative, f(x))

class FundamentalAlgebra:
    """Implementation of Fundamental Theorem of Algebra"""
    
    @staticmethod
    def find_root(coeffs: List[float], max_iter: int = 100) -> complex:
        """
        Find one root of polynomial using Newton's method
        coeffs: list of coefficients [an, an-1, ..., a1, a0] for polynomial anx^n + ... + a1x + a0
        """
        # Create polynomial and its derivative
        p = np.poly1d(coeffs)
        p_prime = p.deriv()
        
        # Start at random point
        z = complex(np.random.rand(), np.random.rand())
        
        for _ in range(max_iter):
            z_new = z - p(z) / p_prime(z)
            if abs(z_new - z) < 1e-10:
                return z_new
            z = z_new
            
        return z
    
    @staticmethod
    def find_all_roots(coeffs: List[float]) -> List[complex]:
        """Find all roots of polynomial using successive division"""
        roots = []
        p = np.poly1d(coeffs)
        
        while len(roots) < len(coeffs) - 1:
            root = FundamentalAlgebra.find_root(p.coefficients)
            roots.append(root)
            p = np.poly1d(p.coefficients) // np.poly1d([1, -root])
            
        return roots

class ProjectiveGeometry:
    """Implementation of concepts from projective geometry"""
    
    @staticmethod
    def cross_ratio(points: List[complex]) -> complex:
        """
        Compute cross-ratio of four points
        Cross ratio is invariant under projective transformations
        """
        if len(points) != 4:
            raise ValueError("Cross ratio requires exactly 4 points")
            
        z1, z2, z3, z4 = points
        return ((z1 - z3) * (z2 - z4)) / ((z1 - z4) * (z2 - z3))
    
    @staticmethod
    def projective_transform(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
        """Apply projective transformation to point"""
        if matrix.shape != (3, 3):
            raise ValueError("Projective transform matrix must be 3x3")
            
        # Convert to homogeneous coordinates
        homogeneous = np.append(point, 1)
        transformed = matrix @ homogeneous
        
        # Convert back from homogeneous coordinates
        return transformed[:2] / transformed[2]

def example_usage():
    # Example: Find roots of x^3 - 1 = 0
    coeffs = [1, 0, 0, -1]  # x^3 - 1
    roots = FundamentalAlgebra.find_all_roots(coeffs)
    print("Cube roots of unity:", roots)
    
    # Example: Verify Cauchy-Riemann for f(z) = z^2
    def u(x, y): return x**2 - y**2  # Real part
    def v(x, y): return 2*x*y        # Imaginary part
    point = 1 + 1j
    is_analytic = ComplexAnalysis.cauchy_riemann_check(u, v, point)
    print(f"z^2 is analytic at 1+i: {is_analytic}")
    
    # Example: Compute residue at simple pole
    def f(z): return 1/(z - 1)
    res = ComplexAnalysis.residue(f, 1)
    print(f"Residue of 1/(z-1) at z=1: {res}")

if __name__ == "__main__":
    example_usage()

@dataclass
class Singularity:
    """Represents a singularity of a complex function"""
    point: complex
    type: str  # 'pole', 'essential', 'removable', 'branch'
    order: Optional[int] = None  # For poles

class ComplexAnalysis:
    """Comprehensive implementation of single-variable complex analysis"""
    
    @staticmethod
    def estimation_lemma(f: Callable, center: complex, radius: float, 
                        M: float = None) -> float:
        """
        ML-Estimation Lemma: If |f(z)| ≤ M on |z-a| = R,
        then |f^(n)(a)/n!| ≤ M/R^n
        """
        if M is None:
            # Compute maximum on circle if not provided
            theta = np.linspace(0, 2*np.pi, 1000)
            z = center + radius * np.exp(1j * theta)
            M = max(abs(f(z_val)) for z_val in z)
        
        def nth_derivative(n: int) -> complex:
            h = 1e-7
            circle = center + h * np.exp(2j * np.pi * np.linspace(0, 1, 1000))
            integral = sum(f(z) / (z - center)**(n+1) * h * np.exp(2j * np.pi * k/1000)
                         for k, z in enumerate(circle))
            return integral / (2j * np.pi * factorial(n))
        
        derivatives = [nth_derivative(n) for n in range(10)]  # First 10 derivatives
        bounds = [M / radius**n for n in range(10)]
        
        return all(abs(derivatives[n]) <= bounds[n] for n in range(10))

    @classmethod
    def residue_theorem(cls, f: Callable, contour: List[complex], 
                       singularities: List[Singularity]) -> complex:
        """
        Compute ∮f(z)dz using the residue theorem
        Returns 2πi times sum of residues
        """
        total_residue = 0
        
        for sing in singularities:
            if cls.point_inside_contour(sing.point, contour):
                residue = cls.compute_residue(f, sing)
                total_residue += residue
        
        return 2 * np.pi * 1j * total_residue

    @staticmethod
    def infinite_product(terms: Callable, n_terms: int) -> complex:
        """
        Compute infinite product Π(1 + aₙ) up to n terms
        terms: function that returns aₙ given n
        """
        product = 1
        for n in range(1, n_terms + 1):
            term = 1 + terms(n)
            if abs(term) < 1e-10:  # Check for convergence
                break
            product *= term
        return product

    @staticmethod
    def analytical_continuation(f: Callable, path: List[complex], 
                              start_val: complex) -> List[complex]:
        """
        Perform analytical continuation along a given path
        Returns function values along the path
        """
        values = [start_val]
        
        for i in range(1, len(path)):
            # Use Taylor series to continue function
            z0, z = path[i-1], path[i]
            h = z - z0
            
            # Compute derivatives at previous point
            derivatives = []
            for n in range(10):  # Use 10 terms
                dz = 1e-7
                circle = z0 + dz * np.exp(2j * np.pi * np.linspace(0, 1, 1000))
                integral = sum(f(w) / (w - z0)**(n+1) * dz * np.exp(2j * np.pi * k/1000)
                             for k, w in enumerate(circle))
                derivatives.append(integral / (2j * np.pi * factorial(n)))
            
            # Compute new value using Taylor series
            value = sum(derivatives[n] * h**n / factorial(n) for n in range(10))
            values.append(value)
        
        return values

    @staticmethod
    def branch_cut(f: Callable, start: complex, end: complex, 
                  n_points: int = 100) -> np.ndarray:
        """
        Create a branch cut between two points
        Returns points along the branch cut
        """
        t = np.linspace(0, 1, n_points)
        cut_points = start + (end - start) * t
        
        # Compute function values on both sides of the cut
        epsilon = 1e-10
        upper_values = [f(z + epsilon*1j) for z in cut_points]
        lower_values = [f(z - epsilon*1j) for z in cut_points]
        
        return cut_points, upper_values, lower_values

    @staticmethod
    def argument_principle(f: Callable, f_prime: Callable, 
                         contour: List[complex]) -> Tuple[int, int]:
        """
        Use argument principle to count zeros and poles
        Returns (number of zeros, number of poles)
        """
        # Compute integral of f'/f
        integral = 0
        for i in range(len(contour)-1):
            z1, z2 = contour[i], contour[i+1]
            
            def integrand(t):
                z = z1 + t*(z2-z1)
                return f_prime(z)/f(z)
            
            integral += quad(lambda x: integrand(x).real, 0, 1)[0]
            integral += 1j * quad(lambda x: integrand(x).imag, 0, 1)[0]
        
        winding_number = int(round(integral.real / (2*np.pi)))
        return winding_number

    @staticmethod
    def laurent_series(f: Callable, center: complex, radius: float, 
                      n_terms: int) -> Tuple[List[complex], List[complex]]:
        """
        Compute Laurent series coefficients around a point
        Returns (negative_coeffs, positive_coeffs)
        """
        def compute_coefficient(n: int) -> complex:
            theta = np.linspace(0, 2*np.pi, 1000)
            z = center + radius * np.exp(1j * theta)
            integrand = f(z) / (z - center)**(n+1)
            integral = np.trapz(integrand, theta) * radius * 1j
            return integral / (2*np.pi*1j)
        
        negative_coeffs = [compute_coefficient(n) for n in range(-n_terms, 0)]
        positive_coeffs = [compute_coefficient(n) for n in range(n_terms)]
        
        return negative_coeffs, positive_coeffs

    @staticmethod
    def cauchy_integral_theorem(f: Callable, contour: List[complex]) -> complex:
        """
        Verify Cauchy's Integral Theorem for a given contour
        Returns the contour integral
        """
        integral = 0
        
        for i in range(len(contour)-1):
            z1, z2 = contour[i], contour[i+1]
            
            def integrand(t):
                z = z1 + t*(z2-z1)
                return f(z) * (z2-z1)
            
            integral += quad(lambda x: integrand(x).real, 0, 1)[0]
            integral += 1j * quad(lambda x: integrand(x).imag, 0, 1)[0]
        
        return integral

    @classmethod
    def analyze_singularity(cls, f: Callable, point: complex) -> Singularity:
        """
        Analyze type of singularity at a point
        Returns Singularity object with classification
        """
        # Check for pole vs essential singularity
        try:
            # Compute Laurent series
            neg_coeffs, _ = cls.laurent_series(f, point, 1e-5, 10)
            
            if all(abs(c) < 1e-10 for c in neg_coeffs):
                return Singularity(point, 'removable')
            elif sum(1 for c in neg_coeffs if abs(c) > 1e-10) < float('inf'):
                order = next(i for i, c in enumerate(neg_coeffs) if abs(c) > 1e-10)
                return Singularity(point, 'pole', order)
            else:
                return Singularity(point, 'essential')
        except:
            return Singularity(point, 'branch')

    @staticmethod
    def point_inside_contour(point: complex, contour: List[complex]) -> bool:
        """Helper method to check if point lies inside contour"""
        winding_number = 0
        for i in range(len(contour)):
            z1 = contour[i]
            z2 = contour[(i+1) % len(contour)]
            if (z1.imag <= point.imag < z2.imag or 
                z2.imag <= point.imag < z1.imag):
                if point.real < max(z1.real, z2.real):
                    if z1.real != z2.real:
                        x_intersect = (z1.real + (point.imag - z1.imag) * 
                                     (z2.real - z1.real) / 
                                     (z2.imag - z1.imag))
                        if point.real < x_intersect:
                            winding_number += 1
        return winding_number % 2 == 1

def example_usage():
    """Demonstrate usage of ComplexAnalysis class"""
    
    # Example function with various singularities
    def f(z):
        return 1/(z**2 + 1)
    
    # Compute residues at singularities
    singularities = [
        Singularity(1j, 'pole', 1),
        Singularity(-1j, 'pole', 1)
    ]
    
    # Define a contour (circle of radius 2)
    theta = np.linspace(0, 2*np.pi, 100)
    contour = [2*np.exp(1j*t) for t in theta]
    
    # Apply residue theorem
    integral = ComplexAnalysis.residue_theorem(f, contour, singularities)
    print(f"Contour integral by residue theorem: {integral:.4f}")
    
    # Compute Laurent series around i
    neg_coeffs, pos_coeffs = ComplexAnalysis.laurent_series(f, 1j, 0.5, 5)
    print("Laurent series coefficients around i:")
    print(f"Negative powers: {neg_coeffs}")
    print(f"Positive powers: {pos_coeffs}")

if __name__ == "__main__":
    example_usage()

@dataclass
class ComplexFunction:
    """Represents a complex function in terms of its real and imaginary parts"""
    u: Callable[[float, float], float]  # Real part
    v: Callable[[float, float], float]  # Imaginary part
    
    def __call__(self, z: complex) -> complex:
        """Evaluate function at complex point z"""
        return self.u(z.real, z.imag) + 1j * self.v(z.real, z.imag)
    
    def partial_derivatives(self, x: float, y: float, h: float = 1e-7) -> Tuple[float, float, float, float]:
        """
        Compute partial derivatives at point (x,y)
        Returns (u_x, u_y, v_x, v_y)
        """
        u_x = (self.u(x + h, y) - self.u(x, y)) / h
        u_y = (self.u(x, y + h) - self.u(x, y)) / h
        v_x = (self.v(x + h, y) - self.v(x, y)) / h
        v_y = (self.v(x, y + h) - self.v(x, y)) / h
        
        return u_x, u_y, v_x, v_y

class CauchyAnalysis:
    """Implementation of Cauchy-Riemann equations and Cauchy's integral theorem"""
    
    @staticmethod
    def check_cauchy_riemann(f: ComplexFunction, point: complex, 
                            tolerance: float = 1e-10) -> bool:
        """
        Check if Cauchy-Riemann equations are satisfied at a point
        ∂u/∂x = ∂v/∂y and ∂u/∂y = -∂v/∂x
        """
        x, y = point.real, point.imag
        u_x, u_y, v_x, v_y = f.partial_derivatives(x, y)
        
        return (np.abs(u_x - v_y) < tolerance and 
                np.abs(u_y + v_x) < tolerance)
    
    @staticmethod
    def visualize_cr_equations(f: ComplexFunction, 
                             region: Tuple[float, float, float, float],
                             grid_size: int = 20) -> None:
        """
        Visualize where Cauchy-Riemann equations are satisfied in a region
        region: (x_min, x_max, y_min, y_max)
        """
        x_min, x_max, y_min, y_max = region
        x = np.linspace(x_min, x_max, grid_size)
        y = np.linspace(y_min, y_max, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Compute CR equation differences
        cr_satisfied = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                point = complex(X[i,j], Y[i,j])
                cr_satisfied[i,j] = CauchyAnalysis.check_cauchy_riemann(f, point)
        
        plt.figure(figsize=(10, 8))
        plt.pcolormesh(X, Y, cr_satisfied, cmap='RdYlGn')
        plt.colorbar(label='C-R Equations Satisfied')
        plt.title('Regions where Cauchy-Riemann Equations are Satisfied')
        plt.xlabel('Re(z)')
        plt.ylabel('Im(z)')
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def cauchy_integral(f: ComplexFunction, contour: Callable[[float], complex], 
                       t_range: Tuple[float, float]) -> complex:
        """
        Compute ∮ f(z)dz along parametric contour
        contour: function mapping parameter t to complex points
        t_range: (t_start, t_end) for parameter
        """
        t_start, t_end = t_range
        
        def integrand_real(t):
            z = contour(t)
            dz = (contour(t + 1e-7) - z) / 1e-7
            return (f(z) * dz).real
            
        def integrand_imag(t):
            z = contour(t)
            dz = (contour(t + 1e-7) - z) / 1e-7
            return (f(z) * dz).imag
        
        real_part = quad(integrand_real, t_start, t_end)[0]
        imag_part = quad(integrand_imag, t_start, t_end)[0]
        
        return real_part + 1j * imag_part
    
    @staticmethod
    def verify_cauchy_theorem(f: ComplexFunction, 
                            contours: List[Callable[[float], complex]],
                            t_ranges: List[Tuple[float, float]],
                            tolerance: float = 1e-10) -> bool:
        """
        Verify Cauchy's integral theorem for a collection of contours
        forming a closed path
        """
        total_integral = 0
        
        for contour, t_range in zip(contours, t_ranges):
            integral = CauchyAnalysis.cauchy_integral(f, contour, t_range)
            total_integral += integral
            
        return abs(total_integral) < tolerance
    
    @staticmethod
    def visualize_contour_integral(f: ComplexFunction, 
                                 contour: Callable[[float], complex],
                                 t_range: Tuple[float, float],
                                 n_points: int = 100) -> None:
        """
        Visualize the contour and integrand for a contour integral
        """
        t_start, t_end = t_range
        t = np.linspace(t_start, t_end, n_points)
        
        # Compute contour points
        points = np.array([contour(ti) for ti in t])
        
        # Compute function values along contour
        values = np.array([f(z) for z in points])
        
        plt.figure(figsize=(15, 5))
        
        # Plot contour
        plt.subplot(121)
        plt.plot(points.real, points.imag, 'b-', label='Contour')
        plt.scatter(points.real, points.imag, c=t, cmap='viridis')
        plt.colorbar(label='Parameter t')
        plt.title('Integration Contour')
        plt.xlabel('Re(z)')
        plt.ylabel('Im(z)')
        plt.grid(True)
        plt.axis('equal')
        
        # Plot integrand
        plt.subplot(122)
        plt.plot(values.real, values.imag, 'r-', label='f(z)')
        plt.scatter(values.real, values.imag, c=t, cmap='viridis')
        plt.colorbar(label='Parameter t')
        plt.title('Function Values along Contour')
        plt.xlabel('Re(f(z))')
        plt.ylabel('Im(f(z))')
        plt.grid(True)
        plt.axis('equal')
        
        plt.tight_layout()
        plt.show()

def example_usage():
    """Demonstrate usage with example functions"""
    
    # Example 1: Analytic function f(z) = z²
    f1 = ComplexFunction(
        u=lambda x, y: x**2 - y**2,  # Real part
        v=lambda x, y: 2*x*y         # Imaginary part
    )
    
    # Check C-R equations at a point
    point = 1 + 1j
    is_analytic = CauchyAnalysis.check_cauchy_riemann(f1, point)
    print(f"z² is analytic at 1+i: {is_analytic}")
    
    # Visualize C-R equations in region
    CauchyAnalysis.visualize_cr_equations(f1, (-2, 2, -2, 2))
    
    # Example 2: Non-analytic function f(z) = conj(z)
    f2 = ComplexFunction(
        u=lambda x, y: x,    # Real part
        v=lambda x, y: -y    # Imaginary part
    )
    
    # Define circular contour
    def circle(t):
        return np.exp(2j * np.pi * t)
    
    # Compute and visualize contour integral
    integral = CauchyAnalysis.cauchy_integral(f2, circle, (0, 1))
    print(f"∮ conj(z)dz around unit circle = {integral:.4f}")
    
    CauchyAnalysis.visualize_contour_integral(f2, circle, (0, 1))
    
    # Verify Cauchy's theorem for f1 (should be nearly zero for analytic function)
    integral_analytic = CauchyAnalysis.cauchy_integral(f1, circle, (0, 1))
    print(f"∮ z²dz around unit circle = {integral_analytic:.4e}")

if __name__ == "__main__":
    example_usage()
