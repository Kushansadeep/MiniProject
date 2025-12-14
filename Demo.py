import pandas as pd
from math import factorial
import numpy as np

# --- 1. M/M/c Analytical Function ---

def calculate_mmc_metrics(lmbda, mu, s, scenario_name):
    """
    Calculates M/M/c queuing metrics analytically.
    lmbda: arrival rate (lambda)
    mu: service rate (mu)
    s: number of servers (capacity)
    """
    rho = lmbda / (s * mu)
    
    # Check for stability (traffic intensity must be < 1 for steady state)
    if rho >= 1:
        return {
            'Scenario': scenario_name,
            'Capacity': s,
            'Load_Type': 'UNSTABLE',
            'Service_Time_Mean': 1/mu,
            'Avg_Wait_Min': float('inf'),
            'Avg_Wait_Min_Change': float('inf'),
            'Throughput_Cust_Per_Min': lmbda
        }

    # Calculate P0 (Probability of 0 customers in the system)
    sum_term1 = 0
    for n in range(s):
        sum_term1 += (lmbda / mu)**n / factorial(n)
        
    term2 = (lmbda / mu)**s / factorial(s) * (1 / (1 - rho))
    p0 = 1 / (sum_term1 + term2)
    
    # Calculate Pw (Probability an arriving customer must wait - P_queue)
    # The M/M/c formula for Pw is: P0 * (lambda/mu)^s / (s! * (1 - rho))
    pw = (lmbda / mu)**s / (factorial(s) * (1 - rho)) * p0
    
    # Calculate Wq (Average time spent in the queue)
    # The M/M/c formula for Wq is: Pw / (s * mu - lambda)
    wq = pw / (s * mu - lmbda)
    
    # Calculate Throughput (lambda in steady state)
    throughput = lmbda
    
    return {
        'Scenario': scenario_name,
        'Capacity': s,
        'Load_Type': 'STABLE',
        'Service_Time_Mean': 1/mu,
        'Avg_Wait_Min': wq,
        'Traffic_Intensity_Rho': rho,
        'Throughput_Cust_Per_Min': throughput
    }

# --- 2. Data Loading and Real Rate Calculation ---

file_name = "dataset.csv"

# Load data and calculate real rates
df = pd.read_csv(file_name)
df['ArrivalTime'] = pd.to_datetime(df['ArrivalTime'])

# Calculate Mu (Service Rate)
mean_service_time_min = df['ServiceMin'].mean()
MU_REAL = 1.0 / mean_service_time_min

# Calculate Lambda (Arrival Rate)
df_sorted = df.sort_values(by='ArrivalTime').reset_index(drop=True)
df_sorted['InterArrivalDuration'] = df_sorted['ArrivalTime'].diff().dt.total_seconds() / 60
mean_inter_arrival_time_min = df_sorted['InterArrivalDuration'].dropna().mean()
LMBDA_REAL = 1.0 / mean_inter_arrival_time_min

# --- 3. Scenario Configuration (Rates) ---

# Stress Test / Peak Load Rates (Used for Figure 4 Analysis)
LMBDA_PEAK = 1.0 / 2.0  # Assumes an average inter-arrival time of 2.0 min at peak
MU_PEAK = 1.0 / 5.0    # Assumes an average service time of 5.0 min at peak

# Faster Service Rate for Peak Load (15% faster service time)
MU_PEAK_FASTER = 1.0 / (5.0 * 0.85)


# --- 4. Scenario Execution ---
# Normal Load (Validation/Section 6)
results_real1 = calculate_mmc_metrics(lmbda=LMBDA_REAL, mu=MU_REAL, s=3, scenario_name="Current System (M/M/3) - Observed Load")
results_real2 = calculate_mmc_metrics(lmbda=LMBDA_REAL, mu=MU_REAL, s=4, scenario_name="Additional Counter (M/M/4) - Observed Load")

# Peak Load (Stress Test/Section 7)
results_peak1 = calculate_mmc_metrics(lmbda=LMBDA_PEAK, mu=MU_PEAK, s=3, scenario_name="Current System (M/M/3) - Peak Load")
results_peak2 = calculate_mmc_metrics(lmbda=LMBDA_PEAK, mu=MU_PEAK, s=4, scenario_name="Additional Counter (M/M/4) - Peak Load")
results_peak3 = calculate_mmc_metrics(lmbda=LMBDA_PEAK, mu=MU_PEAK_FASTER, s=3, scenario_name="Faster Service (M/M/3 + 15% Speed) - Peak Load")


# --- 5. Results Processing ---

# Process Real Load Results
df_real = pd.DataFrame([results_real1, results_real2])
base_wait_real = results_real1['Avg_Wait_Min']
df_real['Avg_Wait_Min_Change'] = (df_real['Avg_Wait_Min'] - base_wait_real) / base_wait_real * 100
# Handle division by zero for negligible wait times
df_real['Avg_Wait_Min_Change'] = np.where(base_wait_real == 0, 0, df_real['Avg_Wait_Min_Change'])

# Process Peak Load Results (Figure 4)
df_peak = pd.DataFrame([results_peak1, results_peak2, results_peak3])
base_wait_peak = results_peak1['Avg_Wait_Min']
df_peak['Avg_Wait_Min_Change'] = (df_peak['Avg_Wait_Min'] - base_wait_peak) / base_wait_peak * 100


# --- 6. Manual Table Printing (Fixing the 'tabulate' error) ---

print("\n" + "="*80)
print("SYSTEM RATES BASED ON REAL DATA")
print("="*80)
print(f"{'Mean Service Time (1/mu)':<40}: {mean_service_time_min:.4f} min")
print(f"{'Mean Inter-Arrival Time (1/lambda)':<40}: {mean_inter_arrival_time_min:.4f} min")
print(f"{'REAL ARRIVAL RATE (Lambda_Real)':<40}: {LMBDA_REAL:.4f} customers/min")
print(f"{'REAL SERVICE RATE (Mu_Real)':<40}: {MU_REAL:.4f} customers/min")
print("="*80)

def print_table(df, title):
    """Prints a DataFrame using standard print formatting."""
    print("\n" + "="*80)
    print(title)
    print("="*80)
    
    # Headers
    headers = ['Scenario', 'Cap.', 'Rho (p)', 'Avg Wait (min)', 'Wait Change (%)']
    header_line = f"{headers[0]:<40} | {headers[1]:<5} | {headers[2]:<8} | {headers[3]:<16} | {headers[4]:<15}"
    print(header_line)
    print("-" * 80)
    
    # Data Rows
    for index, row in df.iterrows():
        rho_str = f"{row['Traffic_Intensity_Rho']:.4f}"
        wait_str = f"{row['Avg_Wait_Min']:.4f}"
        change_str = f"{row['Avg_Wait_Min_Change']:.2f}%"
        
        print(f"{row['Scenario']:<40} | {int(row['Capacity']):<5} | {rho_str:<8} | {wait_str:<16} | {change_str:<15}")

    print("="*80)

# Print Table 1
print_table(df_real, "TABLE 1: SYSTEM PERFORMANCE UNDER NORMAL (OBSERVED) LOAD (Section 6 Validation)")

# Print Table 2 (Figure 4 Data)
print_table(df_peak, "TABLE 2: SYSTEM PERFORMANCE UNDER PEAK LOAD (STRESS TEST - FIGURE 4 DATA)")