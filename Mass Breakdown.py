import numpy as np
import matplotlib.pyplot as plt
# -------------------------
# User inputs & parameters
# -------------------------
iter_max = 200
tol = 0.1          # convergence tolerance for |M_old - M_new|
M_guess = 100  # initial guess for total mass (kg)
M_old = M_guess
payload = 90      # kg

# -------------------------
# Global parameters
# -------------------------
g = 9.81                     # gravity m/s^2
rho = 0.81             # air density kg/m^3
FM = 0.75
eta_prop = 0.85
SoC_min = 0.20
SED = 250                # Wh/Kg
eta_b = 0.85
N_mot = 8
N_prop = 8
PM = 0.5
cD_PL = 0.04353
cL_PL = 1.5
AR = 7.0
e_oswald = 0.85

# geometry/mission
V_cr = 64          # m/s
V_v = 2                    # vertical climb speed m/s
disk_area_per_rotor = 5    # m^2 per rotor
N_rotors = 4
A_total = disk_area_per_rotor * N_rotors
b=6.3                     #wing span m
S_ref = b**2/AR                # wing area m^2
range_m = 60000         # mission range (m)
hover_ht = 20              # used to estimate hover time (m)
climb_ht = 500           # climb altitude (m)
M,Mbatt,Mw,Mfu,Mht,Mvt,Mmot,Mprop,Mlg,ratio_V,I=[],[] ,[] ,[],[] ,[] ,[],[] ,[] ,[],[]
m_old=[]      #list of all M values
pow_hv,pow_cb,pow_cr=[],[],[]
# -------------------------
# Component mass functions(inputs in FPS system , power in hp, out put is in Kg)
# -------------------------
def m_fuselage(total_mass, l_f=19.35, P_max=8.2, Npax=2):
    total_mass=total_mass*2.2046
    return (14.86 * (total_mass**0.144) * ((l_f / P_max)**0.778 )* (l_f**0.383) * (Npax**0.455))*0.4535

def m_wing(total_mass, S=S_ref, eta_w=5, AR=AR):
    total_mass=total_mass*2.2046
    return (0.04674 * (total_mass**0.397) * ((S*3.2808)**0.360) * (eta_w**0.397) * (AR**1.712))*0.4535

def m_tail_h(total_mass, S_th=3.6, AR_th=3.26, trh=0.246):
    total_mass=total_mass*2.2046
    return ((3.184 * (total_mass**0.887) * (S_th**0.101) * (AR_th**0.101)) / (174.04 * (trh**0.223)))*0.4535

def m_tail_v(total_mass, S_tv=1.377, AR_tv=3.75, trv=0.082, Lambda=0):
    total_mass=total_mass*2.2046
    return ((1.68 * (total_mass**0.567) * (S_tv**1.249) * (AR_tv**0.482)) / (639.95 * (trv**0.747) * (np.cos(Lambda)**0.882)))*0.4535

def m_lg(total_mass, llg=2.13, eta_lg=5):
    total_mass=total_mass*2.2046
    return (0.054 * (llg**0.501) * ((total_mass * eta_lg) ** 0.684))*0.4535

def m_prop(P_cb, d_prop=3.28, Nprop=N_prop, N_bl=3):
    return (0.144 * ((d_prop * (P_cb*0.00134) / max(1, Nprop) * (N_bl**0.5)) ** 0.782))*0.4535

# -------------------------
# Power (physics) functions(input in SI units, output in Watt)
# -------------------------

def m_motor(P_cb, PM=PM, Nmot=N_mot):
    return 0.165 *( P_cb*10**-3) * (1 + PM) #mass of all the motors

def induced_velocity_hover(mass, rho_local=rho, A=A_total):
    T = mass * g
    return np.sqrt(T / (2.0 * rho_local * A))

def P_hover(mass, FM=FM, rho_local=rho, A=A_total):
    T = mass * g
    return (T**1.5) / (FM * np.sqrt(2.0 * rho_local * A))

def P_climb(P_hv, v_hv, Vv=V_v):
    # Use supplied climb form from your code
    ratio = Vv / (2.0 * v_hv )
    return P_hv * (ratio + np.sqrt(ratio**2 + 1.0))

def P_cruise(Vc=V_cr, S=S_ref, cD=cD_PL, rho_local=rho, eta_p=eta_prop):
    D = 0.5 * rho_local * Vc**2 * S * cD
    Pcr = D * Vc / eta_p
    return Pcr, D

# -------------------------
# Iteration loop (fixed-point)
# -------------------------
i = 1

print("Starting fixed-point iterations...")
while i <= iter_max:
    I.append(i)
    m_old.append(M_old)
    # compute power numbers using current mass estimate (M_old)
    v_hv = induced_velocity_hover(M_old)
     # Decide branch: climb if V_v >= 0 (typical), descent logic would use negative Vv
     # If you want to treat descent specially, set V_v negative or add branch for descent
    P_hv = P_hover(M_old)               # hover power (W)
    pow_hv.append(P_hv)
    P_cb = P_climb(P_hv, v_hv, Vv=V_v)  # climb power (W) — works for positive V_v
    pow_cb.append(P_cb)
    P_cr, D = P_cruise()
    pow_cr.append(P_cr)

     # mission times (s)
    t_cruise = range_m / V_cr
    # estimate hover time: distance / induced velocity (avoid divide by zero)
    t_hover =  hover_ht /  v_hv
    ratio_V.append(-V_v/v_hv)
    t_climb =  climb_ht /  V_v
    
   
     # estimate hover time: distance / induced velocity (avoid divide by zero)
   
    

     # energy in Joules
    E_Wh= ((P_hv * (t_hover * 2.0) + P_cb * t_climb + P_cr * t_cruise))


     # battery mass (kg): E_Wh / (SED * eta_b) times reserve factor (1+SoC_min)
    m_batt = E_Wh * (1.0 + SoC_min) / (SED * eta_b*3600)
    Mbatt.append(m_batt)

     # structural/propulsion masses using current mass estimate
    mw = m_wing(M_old)
    mfus = m_fuselage(M_old)
    mth = m_tail_h(M_old)
    mtv = m_tail_v(M_old)
    mlg = m_lg(M_old)
    mmot = m_motor(P_cb)
    mprop = m_prop(P_cb)
    
    Mw.append(mw)
    Mfu.append(mfus)
    Mht.append(mth)
    Mvt.append(mtv)
    Mlg.append(mlg)
    Mmot.append(mmot)
    Mprop.append(mprop)

    m_airframe = mw + mfus + mth + mtv + mlg
    m_propulsion = mmot + mprop
    M_s=m_airframe+m_propulsion

    m_total = m_batt + m_airframe + m_propulsion+payload
    M.append(m_total)

    #M_final = ( payload)/(1-(M_s/m_total)-(m_batt/m_total))
    M_new=m_total

     # check convergence
    if abs(M_old - M_new) <= tol:
         
         M_final= M_new
         print(f"Converged at iteration {i}")
         print(f"final mass is {M_final}")
         break
    
    #else:
         # update and continue
    M_old = M_new
    i += 1

plt.plot(I,m_old, color="red",label="mass vs iter")
plt.xlabel("iteration")
plt.ylabel("mass (kg)")
plt.legend()
plt.show()

# Plot a Pie Chart showing the mass breakdown
labels = ['Battery', 'Wing', 'Fuselage', 'Horizontal Tail', 'Vertical Tail', 'Landing Gear', 'Motors', 'Propellers', 'Payload']
sizes = [Mbatt[-1], Mw[-1], Mfu[-1], Mht[-1], Mvt[-1], Mlg[-1], Mmot[-1], Mprop[-1], payload]
colors = ['gold', 'lightblue', 'lightgreen', 'lightcoral', 'violet', 'orange', 'cyan', 'magenta', 'lightgrey']
explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0)  # explode the 1st slice (Battery)
plt.figure(figsize=(8, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.title('Mass Breakdown of the Aircraft')
plt.show()
