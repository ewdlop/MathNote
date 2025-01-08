import numpy as np
import scipy.integrate as integrate

def path_integral_distance(path_function, t_start, t_end):
    """
    Calculate path length using path integral.
    
    Parameters:
    path_function (callable): Parametric function r(t) returning (x,y,z)
    t_start (float): Start of parameter interval
    t_end (float): End of parameter interval
    
    Returns:
    float: Total path length
    """
    def path_length_integrand(t):
        # Derivative of path function
        dt = 1e-6  # Small delta for numerical differentiation
        r1 = path_function(t)
        r2 = path_function(t + dt)
        
        # Calculate differential displacement
        dr = np.array(r2) - np.array(r1)
        return np.linalg.norm(dr / dt)
    
    # Numerical integration of path length
    total_distance, _ = integrate.quad(path_length_integrand, t_start, t_end)
    
    return total_distance

# Example: Helical path
def helical_path(t):
    """
    Parametric representation of a helix
    x = r * cos(t)
    y = r * sin(t)
    z = t
    """
    r = 2  # radius
    return [r * np.cos(t), r * np.sin(t), t]

# Calculate distance along helix
distance = path_integral_distance(helical_path, 0, 2*np.pi)
print(f"Path length of helix: {distance}")
