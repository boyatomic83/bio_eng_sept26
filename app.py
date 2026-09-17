import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Streamlit Page Configuration
st.set_page_config(page_title="Bio-Artificial Lung Simulator", layout="wide")

st.title("🫁 Bio-Artificial Lung & Photobioreactor Simulator")
st.markdown("""
This application models real-time gas exchange across a hollow-fiber membrane hybrid lung, 
combining **passive diffusion (Fick's Law)** with **microalgae photosynthetic carbon fixation**.
""")

# Sidebar Controls
st.sidebar.header("⚙️ Simulation Parameters")

Q_BLOOD = st.sidebar.slider("Blood Flow Rate (L/min)", 1.0, 7.0, 5.0, 0.5)
V_REACTOR = st.sidebar.slider("Reactor Chamber Volume (L)", 0.1, 2.0, 0.5, 0.1)
MAX_PHOTOSYNTHESIS = st.sidebar.slider("Photosynthetic Rate (mmol/L/min)", 0.0, 100.0, 45.0, 5.0)
LIGHT_INTENSITY = st.sidebar.slider("Light Intensity (0 = Dark, 1 = Full Light)", 0.0, 1.0, 1.0, 0.1)

# Constants & Physics
K_O2 = 12.0
K_CO2 = 35.0
CO2_CONSUMPTION_RATIO = 1.0
P_O2_GAS, P_CO2_GAS = 160.0, 0.3
RESIDENCE_TIME = V_REACTOR / Q_BLOOD

# ODE System
def clinical_lung_ode(t, y, light):
    pO2, pCO2, ros = max(y[0], 0.001), max(y[1], 0.001), y[2]
    flux_O2 = K_O2 * (P_O2_GAS - pO2)
    flux_CO2 = K_CO2 * (pCO2 - P_CO2_GAS)
    co2_limitation = pCO2 / (pCO2 + 5.0)
    photo_rate = MAX_PHOTOSYNTHESIS * light * co2_limitation

    dpO2_dt = flux_O2 + photo_rate
    dpCO2_dt = -flux_CO2 - (photo_rate * CO2_CONSUMPTION_RATIO)
    dROS_dt = 0.5 * light if light > 0 else -0.1 * ros
    return [dpO2_dt, dpCO2_dt, dROS_dt]

# Run Integration
initial_state = [40.0, 45.0, 0.0]
t_span = (0, RESIDENCE_TIME)
t_eval = np.linspace(0, RESIDENCE_TIME, 200)

sol_dark = solve_ivp(clinical_lung_ode, t_span, initial_state, args=(0.0,), t_eval=t_eval)
sol_light = solve_ivp(clinical_lung_ode, t_span, initial_state, args=(LIGHT_INTENSITY,), t_eval=t_eval)

# Calculate Hill Saturation (% SO2)
P50, HILL_N = 26.6, 2.7
sat_dark = (sol_dark.y[0]**HILL_N) / (P50**HILL_N + sol_dark.y[0]**HILL_N) * 100
sat_light = (sol_light.y[0]**HILL_N) / (P50**HILL_N + sol_light.y[0]**HILL_N) * 100

# Biocompatibility Math
shear_stress = 12.5 * Q_BLOOD
ros_val = 1.2 * (LIGHT_INTENSITY ** 2)

# Display Key Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Transit Time", f"{RESIDENCE_TIME * 60:.2f} sec")
col2.metric("Exit SO2 (Bio-Hybrid)", f"{sat_light[-1]:.2f}%")
col3.metric("Shear Stress", f"{shear_stress:.1f} Dynes/cm²", "SAFE" if shear_stress < 150 else "DANGER")
col4.metric("ROS Toxicity", f"{ros_val:.2f} µM", "SAFE" if ros_val < 5.0 else "DANGER")

# Render Plots
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].plot(sol_dark.t * 60, sat_dark, 'g--', label='Passive Diffusion Only')
ax[0].plot(sol_light.t * 60, sat_light, 'g-', linewidth=2, label='Bio-Enhanced Hybrid')
ax[0].axhline(y=95, color='gray', linestyle=':', label='Target Saturation (95%)')
ax[0].set_title('Blood Oxygen Saturation (% SO2)')
ax[0].set_xlabel('Time inside Device (Seconds)')
ax[0].set_ylabel('Saturation %')
ax[0].grid(True, alpha=0.3)
ax[0].legend()

ax[1].plot(sol_dark.t * 60, sol_dark.y[1], 'r--', label='CO2 Clearance (Dark)')
ax[1].plot(sol_light.t * 60, sol_light.y[1], 'r-', linewidth=2, label='CO2 Clearance (Light ON)')
ax[1].set_title('pCO2 Clearance (mmHg)')
ax[1].set_xlabel('Time inside Device (Seconds)')
ax[1].set_ylabel('pCO2 mmHg')
ax[1].grid(True, alpha=0.3)
ax[1].legend()

st.pyplot(fig)