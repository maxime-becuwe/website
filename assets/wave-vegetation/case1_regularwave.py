import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import solve_ivp


# def regular wave theory

g = 9.81
rho_w = 1000.0
H_0 = 2.0
T = 4.0
omega = 2 * np.pi / T
h = 5.0

def dispersion_relation(k, omega, h):
    return g * k * np.tanh(k * h) - omega**2

k = brentq(dispersion_relation, 1e-8, 10, args=(omega, h))
print('wave length L:', 2 * np.pi / k)
print('Wave number k:', k)
def eta(x,t,H):
    return H/2 * np.cos(k*x - omega*t)

def u(x,z,t, H):
    return (H/2) * omega * np.cosh(k*(z+h)) / np.sinh(k*h) * np.cos(k*x - omega*t)

def u_dot(x,z,t, H):
    return -(H/2) * omega**2 * np.cosh(k*(z+h)) / np.sinh(k*h) * np.sin(k*x - omega*t)


# plot wave
x_eta = np.linspace(0, 20, 100)
t_eta = 0

plt.figure(figsize=(10, 5))
plt.plot(x_eta, eta(x_eta, t_eta, H_0))
plt.title('Wave Surface Elevation')
plt.xlabel('x (m)')
plt.ylabel('eta (m)')








## there is vegetation on the ground, we a constant height of 1 m from x = 0 to x = 10m.
## we consider the vegetation as rigid cylinders with surface density of 100 cylinder per square meter, diameter of 0.1m.  
## The vegetation is located at the bottom of the water column, so we will consider the water depth to be 4 meters in the vegetated area.

C_D = 1.2 # drag coefficient for a vertical cylinder [adimensional]
C_M = 1.0 # inertia coefficient for a vertical cylinder [adimensional]
d = 0.1 # diameter of the cylinder [m]
h_vegetation = 1 # height of the vegetation [m]
n = 100 # number of cylinders per square meter [1/m^2]
L_vegetation = 50.0



# We know the force due to the vegetation on the wave with the Morison equation. 
# we can calculate the impact on navier stokes equations and finally the impact on the wave height and period.

## From EnvMa_CM_2 (Centrale Nantes course), the energy per unit surface in the current case is:
## E = 1/8 * rho_w * g * H**2 

## We want to compute the dissipation due to the vegetation, which can be computed by the following formula:

### power dissipation per unit volume from drag force + inertia force to the fluid
### p_d = morison_equation(z,t) * u(x,z,t) = F_D = 0.5 * rho_vegetation * C_D * d * u_drag**2 +  F_I = rho_vege * C_M * np.pi * (d/2)**2 * u_dot(x, z, t)


def D_v(x, H):
    # we want to compute the dissipation due to the vegetation for a given x position in the vegetated area (0 < x < 10 m)
    # we will then integrate the power dissipation over the water column in the vegetated area
    # and over time for one period of the wave.

    z = np.linspace(-h, -h + h_vegetation, 100)
    t = np.linspace(0, T, 100)

    p_drag = np.zeros((len(z), len(t)))
    p_inertia = np.zeros((len(z), len(t)))

    for i in range(len(z)):
        for j in range(len(t)):

            velocity = u(x, z[i], t[j], H)
            acceleration = u_dot(x, z[i], t[j], H)

            # Force of water on vegetation per unit volume, Morison's equation
            F_D = 0.5 * rho_w * C_D * n * d * abs(velocity) * velocity
            F_I = rho_w * C_M * n * np.pi * (d/2)**2 * acceleration

            # Power extracted from water [W/m³]
            p_drag[i, j] = F_D * velocity
            p_inertia[i, j] = F_I * velocity

    # Average over one wave period
    mean_drag = np.trapezoid(p_drag, t, axis=1) / T
    mean_inertia = np.trapezoid(p_inertia, t, axis=1) / T

    # Integrate over vegetation height
    D_drag = np.trapezoid(mean_drag, z)
    D_inertia = np.trapezoid(mean_inertia, z)

    return D_drag, D_inertia

    
    
x = np.linspace(0, L_vegetation, 200)
D_drag = np.zeros_like(x)
D_inertia = np.zeros_like(x)

for i in range(len(x)):
    D_drag[i], D_inertia[i] = D_v(x[i], H_0)


plt.figure(figsize=(10, 6))
plt.plot(x, D_drag, label='Drag Dissipation')
plt.legend()
plt.xlabel('Position along the vegetated area (m)')
plt.ylabel('Dissipation (W/m²)')

plt.figure(figsize=(10, 6))
plt.plot(x, D_inertia, label='Inertia Dissipation', color='orange') 
plt.legend()
plt.xlabel('Position along the vegetated area (m)')
plt.ylabel('Dissipation (W/m²)')





# E = 1/8 * rho_w * g * H**2 # energy per unit surface stored in the waves 
# to compute the energy loss due to the vegetation, we consider a dx section of the vegetated area, 
# and remove the Dv dx dissipation from the energy flux F = E * c_g, c_g the group velocity of the wave
# c_g = omega / (2 * k) * (1 + 2 * k * h / np.sinh(2 * k * h)) group velocity of the wave
# In this simple case, there is a analytical solution for the wave height evolution along the vegetated area


# Group velocity
cg = (omega / (2*k)) * (
    1 + 2*k*h / np.sinh(2*k*h)
)



# Differential equation
def dH_dx(x, y):

    H_local = y[0]

    # Compute dissipation at the current position and wave height
    D_drag, D_inertia = D_v(x, H_local)

    # Energy conservation
    return [-4 * D_drag / (rho_w * g * cg * H_local)]


# Positions where we evaluate the solution
x_eval = np.linspace(0, L_vegetation, 100)

# Solve the differential equation
solution = solve_ivp(
    dH_dx,
    (0, L_vegetation),
    [H_0],
    t_eval=x_eval,
    method="RK45",
    rtol=1e-6,
    atol=1e-8
)

if not solution.success:
    raise RuntimeError(solution.message)

H_x = solution.y[0]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(solution.t, H_x, label="Wave height H(x)")
plt.xlabel("Position along the vegetated area (m)")
plt.ylabel("Wave height (m)")
plt.legend()
plt.grid()
plt.show()



