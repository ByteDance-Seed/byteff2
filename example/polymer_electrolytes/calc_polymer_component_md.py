import numpy as np

def calculate_md_components(density_g_cm3, dp, r_value, mw_unit, mw_salt, box_length_angstrom=None, volume_angstrom3=None):
    # Constants
    NA = 6.02214076e23  # Avogadro's number
    
    if box_length_angstrom:
        V_ang3 = box_length_angstrom**3
    elif volume_angstrom3:
        V_ang3 = volume_angstrom3
    else:
        raise ValueError("Provide either box_length_angstrom or volume_angstrom3")

    # Convert density from g/cm^3 to g/Angstrom^3
    # 1 cm^3 = 1e24 Angstrom^3
    density_g_ang3 = density_g_cm3 / 1e24

    # Calculate Number of Polymer Chains (N_poly)
    # Using the derived formula: N_poly = (rho * V * NA) / (DP * (MW_unit + r * MW_salt))
    numerator = density_g_ang3 * V_ang3 * NA
    denominator = dp * (mw_unit + r_value * mw_salt)
    
    n_poly = round(numerator / denominator)
    
    # Calculate Ions based on n_poly to maintain stoichiometry
    n_units_total = n_poly * dp
    n_cations = round(n_units_total * r_value)
    n_anions = n_cations  # Assuming 1:1 salt
    
    # Calculate the actual density and final box length after rounding
    mass_total_g = (n_poly * dp * mw_unit + n_cations * mw_salt) / NA
    actual_volume = mass_total_g / density_g_ang3
    final_L = actual_volume**(1/3)

    return {
        "Polymer Chains": n_poly,
        "Cations": n_cations,
        "Anions": n_anions,
        "Adjusted Box Length (Å)": round(final_L, 4),
        "Total atoms approx": "Depends on chemistry"
    }

# Example for PEO/LiTFSI system (MW PEO unit ~44.05, MW LiTFSI ~287.09)
results = calculate_md_components(
    density_g_cm3=1.2, 
    dp=130, 
    r_value=0.08, 
    mw_unit=58, 
    mw_salt=187.07, 
    box_length_angstrom=60
)

for key, val in results.items():
    print(f"{key}: {val}")
