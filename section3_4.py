#!/usr/bin/env python3
"""
Section 3.4: Convergence Analysis with Graphs
"""

import matplotlib.pyplot as plt
import numpy as np
import random
from simulator import loadBalancer

def calculate_theoretical_values():
    """Calculate theoretical values for M/M/1 with λ=2, μ=5"""
    lambda_ = 2
    mu = 5
    rho = lambda_ / mu
    
    # M/M/1 theoretical formulas
    theoretical_avg_time = 1 / (mu - lambda_)  # Average time in system = 1/3 = 0.3333
    theoretical_customers_per_time = lambda_  # Arrival rate = 2
    
    return theoretical_avg_time, theoretical_customers_per_time, rho

def run_convergence_analysis_with_graphs():
    """
    Run simulation for T=10,20,30,...,90,100 and calculate average error over 20 runs
    Create two graphs showing relative error percentage vs T as requested in homework
    """
    
    theoretical_avg_time, theoretical_customers_per_time, rho = calculate_theoretical_values()
    
    print("Section 3.4: Convergence Analysis with Graphs")
    print("=" * 60)
    print(f"Parameters: λ=2, μ=5, N=1000, ρ={rho:.3f}")
    print(f"Theoretical average time in system: {theoretical_avg_time:.4f}")
    print(f"Theoretical customers per time: {theoretical_customers_per_time:.4f}")
    print()
    
    T_values = list(range(10, 101, 10))  # 10, 20, 30, ..., 100
    num_runs = 20  # Average over 20 runs as requested
    
    time_errors = []
    customer_errors = []
    
    print("Running analysis (20 runs per T value)...")
    print("T   | Time Error% | Customer Error%")
    print("-" * 40)
    
    for T in T_values:
        # Run 20 simulations for each T
        time_error_runs = []
        customer_error_runs = []
        
        for run in range(num_runs):
            # Different seed for each run
            seed = 1000 + T * 100 + run
            np.random.seed(seed)
            random.seed(seed)
            
            lb = loadBalancer(2, 5, 1000)
            lb.run(T)
            
            # Calculate metrics
            customers_served = lb.server.customers_served
            if customers_served > 0:
                total_time = lb.server.total_wait_time + lb.server.total_service_time
                avg_time = total_time / customers_served
            else:
                avg_time = 0
            
            customers_per_time = customers_served / T
            
            # Calculate relative errors (%) using the formula from homework
            # Error(x) = |Theoretical_Value(x) - x| / Theoretical_Value(x) * 100
            time_error = abs(theoretical_avg_time - avg_time) / theoretical_avg_time * 100
            customer_error = abs(theoretical_customers_per_time - customers_per_time) / theoretical_customers_per_time * 100
            
            time_error_runs.append(time_error)
            customer_error_runs.append(customer_error)
        
        # Average over 20 runs
        avg_time_error = np.mean(time_error_runs)
        avg_customer_error = np.mean(customer_error_runs)
        
        time_errors.append(avg_time_error)
        customer_errors.append(avg_customer_error)
        
        print(f"{T:3d} | {avg_time_error:10.2f} | {avg_customer_error:12.2f}")
    
    # Create the two required graphs
    plt.figure(figsize=(14, 6))
    
    # Graph 1: Average time in system error (including service)
    plt.subplot(1, 2, 1)
    plt.plot(T_values, time_errors, "b-o", linewidth=2, markersize=6, label="Time Error")
    plt.xlabel("Simulation Time T")
    plt.ylabel("Relative Error (%)")
    plt.title("Relative Error in Average Time in System\n(Including Service)")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max(time_errors) * 1.1)
    
    # Graph 2: Number of customers served error  
    plt.subplot(1, 2, 2)
    plt.plot(T_values, customer_errors, "r-o", linewidth=2, markersize=6, label="Customer Error")
    plt.xlabel("Simulation Time T")
    plt.ylabel("Relative Error (%)")
    plt.title("Relative Error in Number of\nCustomers Served")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max(customer_errors) * 1.1)
    
    plt.tight_layout()
    plt.savefig("section3_4_convergence_graphs.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # Analysis and conclusions
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY:")
    print("=" * 60)
    print(f"Initial error (T=10): Time={time_errors[0]:.2f}%, Customers={customer_errors[0]:.2f}%")
    print(f"Final error (T=100): Time={time_errors[-1]:.2f}%, Customers={customer_errors[-1]:.2f}%")
    
    # Check convergence trend
    time_improvement = time_errors[0] - time_errors[-1]
    customer_improvement = customer_errors[0] - customer_errors[-1]
    
    print(f"\nCONVERGENCE ANALYSIS:")
    if time_improvement > 0:
        print(f"✓ Time errors DECREASE by {time_improvement:.2f}% (convergence)")
    else:
        print(f"⚠ Time errors do NOT decrease consistently")
    
    if customer_improvement > 0:
        print(f"✓ Customer errors DECREASE by {customer_improvement:.2f}% (convergence)")
    else:
        print(f"⚠ Customer errors do NOT decrease consistently")
    
    print(f"\nCONCLUSION FOR SECTION 3.3:")
    print(f"The results {'ARE' if time_improvement > 0 and customer_improvement > 0 else 'are NOT'} consistent with section 3.3 explanation.")
    print(f"Longer simulation times (larger T) lead to more accurate results,")
    print(f"which is consistent with the law of large numbers and central limit theorem.")
    print(f"As T increases, we get more samples and better statistical accuracy.")
    
    print(f"\nGraphs saved as: section3_4_convergence_graphs.png")
    
    return T_values, time_errors, customer_errors

if __name__ == "__main__":
    run_convergence_analysis_with_graphs()
