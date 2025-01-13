import numpy as np
from functools import reduce
from typing import Callable

def demonstrate_integration_dummy():
    """
    Demonstrate how dummy variables work in integration
    """
    # Define two identical functions with different dummy variables
    def integrand_x(x): return x**2
    def integrand_t(t): return t**2
    
    # Integration limits
    a, b = 0, 1
    n_points = 1000
    
    # Using x as dummy variable
    x = np.linspace(a, b, n_points)
    integral_x = np.trapz(integrand_x(x), x)
    
    # Using t as dummy variable
    t = np.linspace(a, b, n_points)
    integral_t = np.trapz(integrand_t(t), t)
    
    print("Integration results:")
    print(f"∫x²dx from 0 to 1 = {integral_x:.6f}")
    print(f"∫t²dt from 0 to 1 = {integral_t:.6f}")

def demonstrate_summation_dummy():
    """
    Demonstrate dummy variables in summations
    """
    n = 10
    
    # Using i as dummy variable
    sum_i = sum(i for i in range(1, n+1))
    
    # Using k as dummy variable
    sum_k = sum(k for k in range(1, n+1))
    
    print("\nSummation results:")
    print(f"Σi from 1 to {n} = {sum_i}")
    print(f"Σk from 1 to {n} = {sum_k}")

def demonstrate_function_dummy():
    """
    Demonstrate dummy variables in function definitions
    """
    # Three identical functions with different dummy variables
    f = lambda x: x**2 + 2*x
    g = lambda t: t**2 + 2*t
    h = lambda u: u**2 + 2*u
    
    # Test value
    test = 3
    
    print("\nFunction results:")
    print(f"f(x) = x² + 2x at x={test}: {f(test)}")
    print(f"g(t) = t² + 2t at t={test}: {g(test)}")
    print(f"h(u) = u² + 2u at u={test}: {h(test)}")

def demonstrate_map_dummy():
    """
    Demonstrate dummy variables in map/filter operations
    """
    numbers = [1, 2, 3, 4, 5]
    
    # Using x as dummy variable
    squares_x = list(map(lambda x: x**2, numbers))
    
    # Using n as dummy variable
    squares_n = list(map(lambda n: n**2, numbers))
    
    print("\nMap results:")
    print(f"map(λx.x², numbers) = {squares_x}")
    print(f"map(λn.n², numbers) = {squares_n}")

def demonstrate_reduction_dummy():
    """
    Demonstrate dummy variables in reduction operations
    """
    numbers = [1, 2, 3, 4]
    
    # Using a,b as dummy variables
    product_ab = reduce(lambda a, b: a*b, numbers)
    
    # Using x,y as dummy variables
    product_xy = reduce(lambda x, y: x*y, numbers)
    
    print("\nReduction results:")
    print(f"reduce(λ(a,b).a*b, numbers) = {product_ab}")
    print(f"reduce(λ(x,y).x*y, numbers) = {product_xy}")

def demonstrate_comprehension_dummy():
    """
    Demonstrate dummy variables in list comprehensions
    """
    # Using different dummy variables in list comprehensions
    squares_x = [x**2 for x in range(5)]
    squares_i = [i**2 for i in range(5)]
    
    print("\nList comprehension results:")
    print(f"[x² for x in range(5)] = {squares_x}")
    print(f"[i² for i in range(5)] = {squares_i}")

def main():
    """Run all demonstrations"""
    demonstrate_integration_dummy()
    demonstrate_summation_dummy()
    demonstrate_function_dummy()
    demonstrate_map_dummy()
    demonstrate_reduction_dummy()
    demonstrate_comprehension_dummy()

if __name__ == "__main__":
    main()
