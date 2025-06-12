import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)
rc('font', family='serif', size=12)
rc('legend', fontsize=12)


def schwarzschild_to_penrose(R, T):
    """
    Converts Schwarzschild coordinates (R, T) to Penrose-Carter coordinates (U, V).
    Uses different formulas for exterior (|R|>2) and interior (|R|<=2).
    """
    
    exterior = np.abs(R) > 2
    sqrt_arg = np.where(exterior, R/2 - 1, 1 - R/2)
    sqrt_term = np.sqrt(sqrt_arg)
    exp_term = np.exp(R/4)

    
    u_hyp = np.where(exterior, np.cosh(T/4), np.sinh(T/4))
    v_hyp = np.where(exterior, np.sinh(T/4), np.cosh(T/4))

    u = sqrt_term * exp_term * u_hyp
    v = sqrt_term * exp_term * v_hyp

    U = 0.5 * (np.arctan(u + v) + np.arctan(u - v))
    V = 0.5 * (np.arctan(u + v) - np.arctan(u - v))
    return U, V

def create_penrose_grid(R_coords, T_coords):
    """
    Creates a grid of Penrose-Carter coordinates from 1D arrays of
    Schwarzschild coordinates.
    """
    R_grid, T_grid = np.meshgrid(R_coords, T_coords, indexing='ij')
    U, V = schwarzschild_to_penrose(R_grid, T_grid)
    return U, V

def draw_diagram_boundary(offset=0.05, line_density=5):
    """
    Draws the static boundaries, horizons, and labels of the Penrose diagram.

    Parameters:
    - offset: float, offset for annotations
    - line_density: int, number of points per line segment
    """
    pi = np.pi
    # Define line segments: (x_vals, y_vals, kwargs)
    segments = [
        (np.linspace(-pi/2, pi/2, line_density), np.zeros(line_density), {}),
        (np.linspace(-pi/4, pi/4, line_density), np.full(line_density, pi/4), {'linestyle': 'dashed'}),
        (np.linspace(-pi/4, pi/4, line_density), np.full(line_density, -pi/4), {'linestyle': 'dashed'}),
        (np.linspace(-pi/4, pi/4, line_density), np.linspace(-pi/4, pi/4, line_density), {}),
        (np.linspace(-pi/4, pi/4, line_density), -np.linspace(-pi/4, pi/4, line_density), {}),
        (np.linspace(0,np.pi/4,5)-np.pi/2,-np.linspace(0,np.pi/4,5), {}),
        (-np.linspace(0, pi/4, line_density) + pi/2, -np.linspace(0, pi/4, line_density), {}),
        (np.linspace(pi/4, pi/2, line_density), np.linspace(pi/4, 0, line_density), {}),
        (np.linspace(-pi/2, -pi/4, line_density), np.linspace(0, pi/4, line_density), {}),
    ]

    # Plot each segment
    for x, y, kw in segments:
        plt.plot(x, y, color='black', **kw)

    # Define annotations: (text, (x, y), kwargs)
    ann = [
        ('$i^+$', ( pi/4,  pi/4 + offset)),
        ('$i^+$', (-pi/4,  pi/4 + offset)),
        ('$i^-$', ( pi/4, -pi/4 - 2*offset)),
        ('$i^-$', (-pi/4, -pi/4 - 2*offset)),
        ('$\mathcal I^+$', (3*pi/8 + offset,  pi/8 + offset)),
        ('$\mathcal I^+$', (-3*pi/8 - offset, pi/8 + offset)),
        ('$\mathcal I^-$', (3*pi/8 + offset, -pi/8 - 2*offset)),
        ('$\mathcal I^-$', (-3*pi/8 - offset,-pi/8 - 2*offset)),
        ('$R=0$', (0, pi/4 + offset)),
        ('$R=2M$', (-pi/8, -pi/8 - offset), {'rotation': 45}),
        ('$R=2M$', ( pi/8 - 3*offset, -pi/8 - 3*offset), {'rotation': -45}),
    ]

    for args in ann:
        text, (x, y) = args[0], args[1]
        kwargs = args[2] if len(args) > 2 else {}
        plt.annotate(text, (x, y), ha='center', **kwargs)


def plot_penrose(U, V, T_coords, Rconst=None, show_legend=False):
    """
    Plots the Penrose diagram grid, optionally with a constant-R contour.
    """
    # Plot constant-T lines
    for idx, T in enumerate(T_coords):
        plt.plot(U[:, idx], V[:, idx], label=f'$T={T:.0f}$')

    # Plot constant-R contour if specified
    if Rconst is not None:
        T_vals = np.linspace(T_coords.min(), T_coords.max(), 200)
        Uc, Vc = schwarzschild_to_penrose(np.full_like(T_vals, Rconst), T_vals)
        plt.plot(Uc, Vc, linestyle=':', color='black')
        _, V0 = schwarzschild_to_penrose(Rconst, 0)
        plt.annotate(f'$R={Rconst}$', (0, V0 - 3*0.05), ha='center')

    # Draw boundaries and annotations
    draw_diagram_boundary()

    plt.axis('equal')
    plt.axis('off')

    if show_legend:
        plt.legend()
    plt.show()
