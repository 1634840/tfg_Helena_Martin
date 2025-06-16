import numpy as np
import pandas as pd

# --- Configuració inicial --- #

postures = {
    "Decubit lateral_Dreta": np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0]),
    "Decubit lateral_Esquerra": np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0]),
    "Inclinat": np.array([1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]),
    "Sedestació": np.array([1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]),
    "Fowler": np.array([1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0])
    
    
}

prova = {
    "Decubit lateral_Dreta1": np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0]),
    "Decubit lateral_Dreta2": np.array([0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0]),
    "Decubit lateral_Esquerra1": np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0]),
    "Decubit lateral_Esquerra2": np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0]),
    "Decubit Supi_Neutre": np.array([1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]),
    "Decubit Supi_Talons": np.array([1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]),
    "Decubit Supi_V a 22+22": np.array([1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]),
    "Decubit Supi_V.D 45 V.E -7": np.array([1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]),
    "Decubit Supi_V.Dret 30": np.array([1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]),
    "Decubit Supi_V.Dret 45": np.array([1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]),
    "Sedestacio_Completa": np.array([0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0]),
    "Sedestacio_Full Fowler": np.array([1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0]),
    "Sedestacio_Low Fowler": np.array([1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]),
    "Sedestacio_Semi-Fowler": np.array([1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]),
    "Descàrrega cames": np.array([1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]),
    "Descàrrega maluc dret": np.array([0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0]),
    "Descàrrega talons": np.array([1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0])

}

# Factors de risc
morphotype_factor = 1.4
condition_factor = 1.4
risk_multiplier =morphotype_factor*condition_factor

# Pesos per color
color_weights = {"VERD": 1, "GROC": 2, "ROIG": 3}

# Assignació aleatòria de pressions inicials realistes
joint_pressures = np.random.randint(40, 60, size=14)

# Estat inicial
cumulative_stress = np.zeros(14, dtype=int)
time_under_pressure = np.zeros(14, dtype=int)
max_tolerated_time = np.zeros(14)
colors = ["VERD"] * 14
minutes = 0
changes = 0
max_changes = 3
current_posture = "Decubit lateral_Dreta"
history = []
change_log = []
posture_history = []
max_history = 3
min_minutes_between_changes = 90
last_change_minute = 0  




def calculate_max_tolerated_time(P0, morphotype_factor, condition_factor):
    """
    Versió simplificada i estable del model Linder-Ganz per a pressions < 195 mmHg.
    """
    K = 157.5
    C = 37.5
    alpha = 0.15
    t0 = 90

    # Protecció de domini
    if P0 <= C or P0 >= K + C:
        return 0  # Pressió fora de rang fisiològic

    # Fórmula inversa de la sigmoide
    ratio = K / (P0 - C)
    delta = ratio - 1

    if delta <= 0:
        return 0  

    m = t0 + (1 / alpha) * np.log(delta)
    m_adjusted = m / (morphotype_factor * condition_factor)

    return max(5, min(m_adjusted, 720))  # Límits pràctics: mínim 5 min, màxim 12h



def get_risk_color(t, max_time):
    if t < max_time :
        return "VERD"
    elif t <= max_time + 15:
        return "GROC"
    else:
        return "ROIG"

def recommend_next_posture(current_posture, cumulative_stress, colors, posture_history):
    min_overlap = float('inf')
    best_posture = None
    for name, vector in postures.items():
        if name == current_posture or name in posture_history:
            continue  
        overlap = np.sum([
            cumulative_stress[i] * vector[i] * color_weights[colors[i]] * risk_multiplier
            for i in range(14)
        ])
        if overlap < min_overlap:
            min_overlap = overlap
            best_posture = name
    return best_posture or current_posture  


# --- Simulació --- #

stress_threshold = 30
time_step = 5
stress_relief = 5

while changes < max_changes:
    previous_posture = current_posture
    cumulative_stress += postures[current_posture] * time_step
    minutes += time_step

    for i in range(14):
        if postures[current_posture][i] == 1:
            time_under_pressure[i] += time_step
            P0 = joint_pressures[i]
            max_tolerated_time[i] = calculate_max_tolerated_time(P0, morphotype_factor, condition_factor)
            colors[i] = get_risk_color(time_under_pressure[i], max_tolerated_time[i])
        else:
            time_under_pressure[i] = 0
            colors[i] = "VERD"

    history.append((minutes, current_posture, cumulative_stress.copy(), colors.copy()))

    if "ROIG" in colors:
        risky_indices = [i for i, c in enumerate(colors) if c == "ROIG"]
        next_posture = recommend_next_posture(current_posture, cumulative_stress, colors, posture_history)
        change_log.append({
            "Minut": minutes,
            "Postura inicial": current_posture,
            "Postura recomanada": next_posture,
            "Colors postura inicial": colors.copy(),
            "Articulacions en ROIG": risky_indices,
            "P₀ (mmHg)": [joint_pressures[i] for i in risky_indices],
            "m' (min)": [round(max_tolerated_time[i], 2) for i in risky_indices]
        })

        released = (postures[current_posture] == 1) & (postures[next_posture] == 0)
        cumulative_stress[released] = 0
        time_under_pressure[released] = 0
        
        posture_history.append(current_posture)
        if len(posture_history) > max_history:
            posture_history.pop(0)
            
        current_posture = next_posture
        last_change_minute = minutes
        changes += 1
        print(f"✔️ Canvi #{changes}: {previous_posture} → {current_posture} a {minutes} minuts")



    else:
        if previous_posture != current_posture:
            print(f"⚠️ ATENCIÓ: la postura ha canviat de {previous_posture} a {current_posture} sense ROIG!")


        

df_risc = pd.DataFrame([
    {
        "Minut": t,
        "Postura": posture,
        "Estrès acumulat": stress.tolist(),
        "Colors de risc": colors
    } for t, posture, stress, colors in history if t % stress_threshold == 0 or "ROIG" in colors
])

df_limits = pd.DataFrame({
    "Índex": list(range(14)),
    "Pressió P₀ (mmHg)": joint_pressures.tolist(),
    "Temps màxim tolerat m' (min)": max_tolerated_time.tolist()
})

df_canvis = pd.DataFrame(change_log)
print("\nColumnes disponibles a df_canvis:")
print(df_canvis.columns.tolist())



print("\n--- Recomanacions de canvi de postura ---")
print(df_canvis[["Minut", "Postura inicial", "Postura recomanada"]].to_string(index=False))


