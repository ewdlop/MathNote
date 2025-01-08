import numpy as np
import sympy as sp
import scipy.optimize as optimize

class HessianMatrixCalculator:
    """
    Comprehensive Hessian Matrix Calculator
    
    Supports multiple methods of Hessian matrix computation:
    1. Symbolic differentiation
    2. Numerical approximation
    3. Automatic differentiation
    4. Optimization-based estimation
    """
    
    @staticmethod
    def symbolic_hessian(func, variables):
        """
        Calculate Hessian matrix using symbolic differentiation
        
        Parameters:
        func (sympy expression): Symbolic function
        variables (list): List of symbolic variables
        
        Returns:
        sympy Matrix: Hessian matrix
        """
        # Compute first derivatives
        first_derivs = [sp.diff(func, var) for var in variables]
        
        # Compute second derivatives (Hessian)
        hessian_matrix = sp.Matrix([
            [sp.diff(deriv, var) for var in variables]
            for deriv in first_derivs
        ])
        
        return hessian_matrix
    
    @staticmethod
    def numerical_hessian(func, x0, method='central'):
        """
        Calculate Hessian matrix numerically
        
        Parameters:
        func (callable): Scalar-valued function
        x0 (numpy array): Point at which to compute Hessian
        method (str): Differentiation method ('forward', 'central', 'complex')
        
        Returns:
        numpy array: Numerical Hessian matrix
        """
        def grad(x):
            """Compute gradient using finite differences"""
            eps = np.sqrt(np.finfo(float).eps)
            grad_vec = np.zeros_like(x, dtype=float)
            
            for i in range(len(x)):
                # Create perturbation vectors
                x_plus = x.copy()
                x_minus = x.copy()
                
                if method == 'forward':
                    x_plus[i] += eps
                    grad_vec[i] = (func(x_plus) - func(x)) / eps
                
                elif method == 'central':
                    x_plus[i] += eps
                    x_minus[i] -= eps
                    grad_vec[i] = (func(x_plus) - func(x_minus)) / (2 * eps)
                
                elif method == 'complex':
                    # Complex step method (more accurate)
                    h = 1e-20j
                    x_complex = x.copy().astype(complex)
                    x_complex[i] += h
                    grad_vec[i] = np.imag(func(x_complex)) / np.imag(h)
            
            return grad_vec
        
        # Hessian via finite differences of gradient
        def hessian_func(x):
            hess = np.zeros((len(x), len(x)))
            eps = np.sqrt(np.finfo(float).eps)
            
            for i in range(len(x)):
                for j in range(len(x)):
                    # Perturb in both directions
                    x_ij_plus = x.copy()
                    x_ij_plus[i] += eps
                    x_ij_plus[j] += eps
                    
                    # Compute mixed partial derivatives
                    hess[i, j] = (
                        func(x_ij_plus) 
                        - func(x_ij_plus - np.array([eps if k==i else 0 for k in range(len(x))])) 
                        - func(x_ij_plus - np.array([eps if k==j else 0 for k in range(len(x))])) 
                        + func(x)
                    ) / (eps**2)
            
            return hess
        
        return hessian_func(x0)
    
    @staticmethod
    def optimization_hessian(func, x0):
        """
        Compute Hessian using optimization library
        
        Parameters:
        func (callable): Objective function
        x0 (numpy array): Initial guess
        
        Returns:
        numpy array: Hessian matrix estimated during optimization
        """
        # Use scipy's optimization to get Hessian
        result = optimize.minimize(
            func, 
            x0, 
            method='BFGS',  # Quasi-Newton method that approximates Hessian
            options={'disp': False}
        )
        
        return result.hess_inv.T  # Inverse Hessian
    
    @staticmethod
    def example_demonstrations():
        """
        Demonstrate Hessian calculation for different functions
        """
        # Symbolic Example: f(x,y) = x^2 + y^2
        print("Symbolic Hessian Demonstration:")
        x, y = sp.symbols('x y')
        symbolic_func = x**2 + y**2
        symbolic_vars = [x, y]
        symbolic_hess = HessianCalculator.symbolic_hessian(symbolic_func, symbolic_vars)
        print("Symbolic Hessian:\n", symbolic_hess)
        
        # Numerical Example
        print("\nNumerical Hessian Demonstration:")
        def numerical_func(x):
            return x[0]**2 + x[1]**2
        
        x0 = np.array([1.0, 2.0])
        
        # Different numerical methods
        methods = ['forward', 'central', 'complex']
        for method in methods:
            print(f"\nNumerical Hessian ({method} difference):")
            print(HessianCalculator.numerical_hessian(numerical_func, x0, method=method))
        
        # Optimization-based Hessian
        print("\nOptimization-based Hessian:")
        print(HessianCalculator.optimization_hessian(numerical_func, x0))

def main():
    # Run demonstrations
    HessianMatrixCalculator.example_demonstrations()

if __name__ == "__main__":
    main()
