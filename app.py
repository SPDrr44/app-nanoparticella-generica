import streamlit as st
import numpy as np
from scipy.optimize import brentq

# --- COSTANTI FISICHE ---
h = 6.62607015e-34          # Costante di Planck (J*s)
c = 2.99792458e8            # Velocità della luce (m/s)
e_charge = 1.602176634e-19  # Carica dell'elettrone (C)
eps0 = 8.8541878128e-12     # Permittività del vuoto (F/m)
m0 = 9.10938356e-31         # Massa a riposo dell'elettrone (kg)
hbar = h / (2 * np.pi)
eV_to_J = e_charge          # 1 eV = 1.602e-19 J
J_to_eV = 1.0 / eV_to_J

# --- FUNZIONE CHE CALCOLA IL GAP DELLA NANOPARTICELLA ---
def Eg_confined_J(R, Eg_bulk_J, me_eff, mh_eff, epsilon):
    """
    Calcola E_g* (in Joule) per una nanoparticella di raggio R,
    secondo il modello di Brus.
    """
    # Termine cinetico
    kinetic = (hbar**2 * (np.pi**2) / (2.0 * R**2)) * (1.0 / (me_eff * m0) + 1.0 / (mh_eff * m0))
    # Termine coulombiano
    coulomb = (1.8 * e_charge**2) / (4.0 * np.pi * eps0 * epsilon * R)
    return Eg_bulk_J + kinetic - coulomb

def solve_for_radius(Eg_target_J, Eg_bulk_J, me_eff, mh_eff, epsilon, R_min, R_max):
    """
    Risolve l'equazione:
    Eg_confined_J(R, ...) = Eg_target_J
    per R nell'intervallo [R_min, R_max] utilizzando il metodo di Brent.
    """
    def f(R):
        return Eg_confined_J(R, Eg_bulk_J, me_eff, mh_eff, epsilon) - Eg_target_J
    return brentq(f, R_min, R_max)

# --- INTERFACCIA UTENTE CON STREAMLIT ---
st.title("Calcolo Raggio Nanoparticella - Modello Generico")

st.markdown("""
Inserisci i dati seguenti:
- **Gap bulk** \(E_{g,\mathrm{bulk}}\) [eV]
- **Massa efficace elettrone** \(m_e^*\) (in multiplo di \(m_0\))
- **Massa efficace lacuna** \(m_h^*\) (in multiplo di \(m_0\))
- **Permittività relativa** \(\epsilon\)
- **Lunghezza d'onda della nanoparticella** \(\lambda_c\) [nm]
""")

# Input utente
Eg_bulk_eV = st.number_input("Gap bulk \(E_{g,\mathrm{bulk}}\) (eV)", value=3.4, min_value=0.0)
me_eff = st.number_input("Massa efficace elettrone \(m_e^*\) (multiplo di \(m_0\))", value=0.24, min_value=0.0)
mh_eff = st.number_input("Massa efficace lacuna \(m_h^*\) (multiplo di \(m_0\))", value=0.59, min_value=0.0)
epsilon = st.number_input("Permittività relativa \(\epsilon\)", value=8.66, min_value=0.0)
lambda_c_nm = st.number_input("Lunghezza d'onda della nanoparticella \(\lambda_c\) (nm)", value=350.0, min_value=0.0)

# Checkbox per mostrare i calcoli intermedi
show_details = st.checkbox("Mostra calcoli intermedi")

if st.button("Calcola Raggio"):
    # Converte il gap bulk in Joule
    Eg_bulk_J = Eg_bulk_eV * eV_to_J

    # Calcola E_g* dalla lunghezza d'onda: E_g* = hc/λ
    lambda_c_m = lambda_c_nm * 1e-9   # converte nm in m
    Eg_target_J = (h * c) / lambda_c_m
    Eg_target_eV = Eg_target_J * J_to_eV

    # Definiamo un intervallo di ricerca per R (es. tra 0.1 nm e 50 nm)
    R_min = 0.1e-9
    R_max = 50e-9

    if Eg_target_J < Eg_bulk_J:
        st.error("L'energia del gap misurato risulta minore del gap bulk. Controlla i dati inseriti!")
    else:
        try:
            R_solution_m = solve_for_radius(Eg_target_J, Eg_bulk_J, me_eff, mh_eff, epsilon, R_min, R_max)
            R_solution_nm = R_solution_m * 1e9
            st.success(f"Il raggio della nanoparticella è: **{R_solution_nm:.3f} nm**")
            gap_calcolato_eV = Eg_confined_J(R_solution_m, Eg_bulk_J, me_eff, mh_eff, epsilon) * J_to_eV
            st.write(f"Gap nanoparticella calcolato: **{gap_calcolato_eV:.3f} eV**")
            
            if show_details:
                st.markdown("### Dettagli dei calcoli intermedi:")
                st.write(f"**Lunghezza d'onda inserita:** {lambda_c_nm:.3f} nm → {lambda_c_m:.3e} m")
                st.write(f"**Gap nanoparticella target:** {Eg_target_eV:.3f} eV (→ {Eg_target_J:.3e} J)")
                st.write(f"**Gap bulk:** {Eg_bulk_eV:.3f} eV (→ {Eg_bulk_J:.3e} J)")
                kinetic = (hbar**2 * (np.pi**2) / (2.0 * R_solution_m**2)) * (1.0/(me_eff*m0) + 1.0/(mh_eff*m0))
                coulomb = (1.8 * e_charge**2) / (4.0 * np.pi * eps0 * epsilon * R_solution_m)
                st.write(f"**Termine cinetico calcolato:** {kinetic:.3e} J")
                st.write(f"**Termine coulombiano calcolato:** {coulomb:.3e} J")
                st.write(f"**Somma (Eg_bulk + termini):** {(Eg_bulk_J + kinetic - coulomb):.3e} J")
                
        except ValueError as e:
            st.error(f"Impossibile trovare una soluzione: {e}")
