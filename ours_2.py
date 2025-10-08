import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# VTOL UAV Fixed-Point Sizing (Powered-Lift)
# =====================================================

# -------------------------
# Iteration settings
# -------------------------
iter_max = 200
tol = 0.1
M_guess = 100  # kg, target UAV mass
M_old = M_guess
payload = 15  # kg

# -------------------------
# Global parameters
# -------------------------
g = 9.81
rho = 0.777  # kg/m³ at 4500 m

FM = 0.75
eta_prop = 0.85
SoC_min = 0.20
SED = 200  # Wh/kg
eta_b = 0.85
N_mot = 4
N_prop = 4
PM = 0.50

# Aerodynamic coefficients
cD_PL = 0.05
cL_PL = 1.5
AR = 8.0
e_oswald = 0.85

# -------------------------
# Geometry / Mission
# -------------------------
V_cr = 20.0       # m/s cruise
V_v = 2.5         # m/s climb rate
hover_ht = 40   # m
climb_ht = 1000    # m (to 5000 m)
range_m = 10000   # 10 km

# Rotor properties
disk_area_per_rotor = 0.6  # m²
N_rotors = 2
A_total = disk_area_per_rotor * N_rotors

# Wing geometry (scaled down)
b = 5.5
S_ref = b**2 / AR

# -------------------------
# Data storage
# -------------------------
I, M_vals, Mbatt, Mw, Mfu, Mht, Mvt, Mlg, Mmot, Mprop = [], [], [], [], [], [], [], [], [], []
pow_hv, pow_cb, pow_cr = [], [], []

# -------------------------
# Component mass models
# -------------------------
def m_fuselage(total_mass, l_f=2.5, P_max=2.0, Npax=0.25):
    total_mass_lb = total_mass * 2.2046
    l_f_ft = l_f * 3.28084
    P_max_ft = P_max * 3.28084
    W_fus_lb = (14.86 * (total_mass_lb ** 0.144) *
                ((l_f_ft / P_max_ft) ** 0.778) *
                (l_f_ft ** 0.383) * (Npax ** 0.455))
    return W_fus_lb * 0.4535

def m_wing(total_mass, S=S_ref, eta_w=5, AR=AR):
    total_mass=total_mass*2.2046
    return (0.04674 * (total_mass**0.397) * (S**0.360) * (eta_w**0.397) * (AR**1.712))*0.4535

def m_tail_h(total_mass, S_th=S_ref*0.25, AR_th=3.26, trh=0.246):
    total_mass = total_mass * 2.2046
    return ((3.184 * (total_mass**0.887) * (S_th**0.101) * (AR_th**0.101))
            / (174.04 * (trh**0.223))) * 0.4535

def m_tail_v(total_mass, S_tv=S_ref*0.15, AR_tv=3.75, trv=0.082, Lambda=0):
    total_mass = total_mass * 2.2046
    return ((1.68 * (total_mass**0.567) * (S_tv**1.249) * (AR_tv**0.482))
            / (639.95 * (trv**0.747) * (np.cos(Lambda)**0.882))) * 0.4535

# def m_lg(total_mass, llg=2.0, eta_lg=5):
#     total_mass = total_mass * 2.2046
#     return (0.054 * (llg**0.501) * ((total_mass * eta_lg)**0.684)) * 0.4535

def m_motor(P_cb):
    return 0.165 * (P_cb * 1e-3) * (1 + PM)

def m_prop(P_cb, d_prop=0.5, Nprop=N_prop, N_bl=4):
    d_prop_ft = d_prop * 3.28084
    W_prop_lb = (0.144 * ((d_prop_ft *
                           (P_cb * 0.00134) /
                           max(1, Nprop) *
                           (N_bl ** 0.5)) ** 0.782))
    return W_prop_lb * 0.4535

# -------------------------
# Power models
# -------------------------
def induced_velocity_hover(mass, rho_local=rho, A=A_total):
    T = mass * g
    return np.sqrt(T / (2.0 * rho_local * A))

def P_hover(mass, FM=FM, rho_local=rho, A=A_total):
    T = mass * g
    return (T**1.5) / (FM * np.sqrt(2.0 * rho_local * A))

def P_climb(P_hv, v_hv, Vv=V_v):
    ratio = Vv / (2.0 * v_hv)
    return P_hv * (ratio + np.sqrt(ratio**2 + 1.0))

def P_cruise(Vc=V_cr, S=S_ref, cD=cD_PL, rho_local=rho, eta_p=eta_prop):
    D = 0.5 * rho_local * Vc**2 * S * cD
    Pcr = D * Vc / eta_p
    return Pcr, D

# -------------------------
# Fixed-point iteration
# -------------------------
print("Starting fixed-point iterations...\n")

for i in range(1, iter_max + 1):
    I.append(i)

    v_hv = induced_velocity_hover(M_old)
    P_hv = P_hover(M_old)
    P_cb = P_climb(P_hv, v_hv, V_v)
    P_cr, D = P_cruise()

    pow_hv.append(P_hv)
    pow_cb.append(P_cb)
    pow_cr.append(P_cr)

    # Mission times (s)
    t_hover = hover_ht / v_hv
    t_climb = climb_ht / V_v
    t_cruise = range_m / V_cr + 25*60  # add 25 min loiter time

    # Total energy (J)
    E_total = P_hv * (t_hover * 2.0) + P_cb * t_climb + P_cr * t_cruise

    # Battery mass (kg)
    m_batt = E_total * (1.0 + SoC_min) / (SED * eta_b * 3600)
    Mbatt.append(m_batt)

    # Structural and propulsion masses
    mw = m_wing(M_old)
    mfus = m_fuselage(M_old)
    mth = m_tail_h(M_old)
    mtv = m_tail_v(M_old)
    #mlg = m_lg(M_old)
    mmot = m_motor(P_cb)
    mprop = m_prop(P_cb)

    Mw.append(mw)
    Mfu.append(mfus)
    Mht.append(mth)
    Mvt.append(mtv)
    #Mlg.append(mlg)
    Mmot.append(mmot)
    Mprop.append(mprop)

    m_airframe = mw + mfus + mth + mtv #+ mlg
    m_propulsion = mmot + mprop

    M_new = m_batt + m_airframe + m_propulsion + payload
    M_vals.append(M_new)

    if abs(M_old - M_new) <= tol:
        print(f"Converged at iteration {i}")
        print(f"Final Takeoff Mass: {M_new:.2f} kg\n")
        break

    M_old = M_new

# -------------------------
# Results Summary
# -------------------------
print("Results Summary:")
print(f"  Fuselage Mass: {Mfu[-1]:.2f} kg")
print(f"  Wing Mass: {Mw[-1]:.2f} kg")
print(f"  Horizontal Tail Mass: {Mht[-1]:.2f} kg")
print(f"  Vertical Tail Mass: {Mvt[-1]:.2f} kg")
#print(f"  Landing Gear Mass: {Mlg[-1]:.2f} kg")
print(f"  Total Airframe Mass: {m_airframe:.2f} kg")
print(f"  Propulsion Mass: {m_propulsion:.2f} kg")
print(f"  Battery Mass: {Mbatt[-1]:.2f} kg")
print(f"  Payload Mass: {payload:.2f} kg")
print(f"  Final Takeoff Mass: {M_new:.2f} kg\n")

# -------------------------
# Convergence Plot
# -------------------------
plt.figure()
plt.plot(I[:len(M_vals)], M_vals, color="red", label="Total mass vs iteration")
plt.xlabel("Iteration")
plt.ylabel("Total Mass (kg)")
plt.title("Fixed-Point Convergence")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# Mass Breakdown Pie Chart
# -------------------------
labels = ['Airframe', 'Propulsion', 'Battery', 'Payload']
airframe_mass = Mw[-1] + Mfu[-1] + Mht[-1] + Mvt[-1] #+ Mlg[-1]
propulsion_mass = Mmot[-1] + Mprop[-1]
sizes = [airframe_mass, propulsion_mass, Mbatt[-1], payload]
colors = ['lightblue', 'lightgreen', 'lightcoral', 'gold']
explode = (0.1, 0.1, 0.1, 0)
plt.figure()
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
# plt.title("Mass Breakdown")
plt.axis('equal')
plt.savefig('our_case.png')
plt.show()
