import numpy as np
import math

def sphere_area(radius):
    """
    Calculate the surface area of a sphere using spherical coordinates.
    
    Surface area in spherical coordinates is calculated by double integration:
    A = ∫∫ r²sin(θ) dθ dφ
    where r is radius, θ is polar angle (0 to π), φ is azimuthal angle (0 to 2π)
    
    Parameters:
    radius (float): Radius of the sphere
    
    Returns:
    float: Total surface area of the sphere
    """
    # Analytical solution: 4πr²
    analytical_area = 4 * math.pi * radius**2
    
    # Numerical integration verification
    def integrand(theta, phi):
        return radius**2 * np.sin(theta)
    
    # Using numpy's numerical integration
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    
    # Meshgrid for integration
    THETA, PHI = np.meshgrid(theta, phi)
    
    # Numerical double integration
    numerical_area = np.trapz(
        np.trapz(integrand(THETA, PHI), phi, axis=0), 
        theta
    )
    
    print(f"Analytical Area: {analytical_area}")
    print(f"Numerical Area: {numerical_area}")
    print(f"Difference: {abs(analytical_area - numerical_area)}")
    
    return analytical_area

# Example usage
r = 5
total_area = sphere_area(r)
print(f"\nTotal surface area of sphere with radius {r}: {total_area}")
