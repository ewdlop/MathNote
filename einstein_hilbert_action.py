import numpy as np
import sympy as sp
from sympy import Matrix, Symbol, diff, simplify, sqrt

# Define spacetime coordinates
t, x, y, z = sp.symbols('t x y z')
coords = [t, x, y, z]

# Define metric components as functions
g_components = [[Symbol(f'g_{i}{j}')(t,x,y,z) for j in range(4)] for i in range(4)]
g = Matrix(g_components)
g_inv = g.inv()

# Compute Christoffel symbols
def christoffel(g, g_inv, coords):
    n = len(coords)
    gamma = np.zeros((n,n,n), dtype=object)
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                sum_term = 0
                for l in range(n):
                    # ∂g_ij/∂x^k + ∂g_ik/∂x^j - ∂g_jk/∂x^i
                    term = (diff(g[i,j], coords[k]) + 
                           diff(g[i,k], coords[j]) - 
                           diff(g[j,k], coords[i]))
                    sum_term += g_inv[l,k] * term/2
                gamma[i,j,k] = simplify(sum_term)
    return gamma

# Compute Riemann tensor
def riemann_tensor(gamma, coords):
    n = len(coords)
    R = np.zeros((n,n,n,n), dtype=object)
    
    for rho in range(n):
        for sigma in range(n):
            for mu in range(n):
                for nu in range(n):
                    # R^rho_sigma,mu,nu = ∂_μΓ^ρ_νσ - ∂_νΓ^ρ_μσ + 
                    # Γ^ρ_μλΓ^λ_νσ - Γ^ρ_νλΓ^λ_μσ
                    term1 = diff(gamma[rho,nu,sigma], coords[mu])
                    term2 = -diff(gamma[rho,mu,sigma], coords[nu])
                    
                    term3 = sum(gamma[rho,mu,l] * gamma[l,nu,sigma] 
                              for l in range(n))
                    term4 = -sum(gamma[rho,nu,l] * gamma[l,mu,sigma] 
                               for l in range(n))
                    
                    R[rho,sigma,mu,nu] = simplify(term1 + term2 + term3 + term4)
    return R

# Compute Ricci tensor
def ricci_tensor(R):
    n = R.shape[0]
    Ric = np.zeros((n,n), dtype=object)
    
    for mu in range(n):
        for nu in range(n):
            # R_μν = R^ρ_μρν
            Ric[mu,nu] = sum(R[rho,mu,rho,nu] for rho in range(n))
    return Ric

# Compute Ricci scalar
def ricci_scalar(Ric, g_inv):
    # R = g^μν R_μν
    R = sum(sum(g_inv[mu,nu] * Ric[mu,nu] 
              for nu in range(4)) for mu in range(4))
    return simplify(R)

# Einstein-Hilbert action
def einstein_hilbert_action(R, g):
    # S = ∫ √(-g) R d⁴x
    det_g = g.det()
    sqrt_minus_g = sqrt(-det_g)
    return sqrt_minus_g * R

# Add matter coupling
def total_action(S_EH, L_matter):
    # S_total = S_EH + S_matter
    kappa = Symbol('kappa')  # κ = 8πG/c⁴
    return S_EH - kappa * L_matter

# Compute metric variation
def metric_variation(S_total, g):
    n = g.shape[0]
    delta_S = np.zeros((n,n), dtype=object)
    
    for mu in range(n):
        for nu in range(n):
            # δS/δg_μν
            delta_S[mu,nu] = diff(S_total, g[mu,nu])
    return delta_S

# Example usage:
gamma = christoffel(g, g_inv, coords)
R_tensor = riemann_tensor(gamma, coords)
Ric_tensor = ricci_tensor(R_tensor)
R_scalar = ricci_scalar(Ric_tensor, g_inv)
S_EH = einstein_hilbert_action(R_scalar, g)

# For vacuum Einstein equations:
L_matter = 0
S_total = total_action(S_EH, L_matter)
equations = metric_variation(S_total, g)
