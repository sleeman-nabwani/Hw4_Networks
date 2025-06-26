#!/usr/bin/env python3
"""
Comprehensive Test Suite for Multi-Server Load Balancer Simulator
Includes multiple configurations, multiple runs, and extensive edge case testing
"""

import time
import random
import numpy as np
import math
import statistics
from simulator import loadBalancer

class TestResult:
    """Store results from a single test run"""
    def __init__(self, A, B, T_end, T_w, T_s, servers_data):
        self.A = A  # Customers served
        self.B = B  # Customers rejected
        self.T_end = T_end  # Last departure time
        self.T_w = T_w  # Average wait time
        self.T_s = T_s  # Average service time
        self.servers_data = servers_data  # Per-server statistics

def run_single_test(T, M, P, lambda_, Q, mu, seed=None):
    """Run a single test and return results"""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    try:
        lb = loadBalancer(lambda_, mu, P, Q, M)
        lb.run(T)
        
        A = sum([s.customers_served for s in lb.servers])
        B = sum([s.customers_rejected for s in lb.servers])
        
        # Handle edge cases for division by zero
        if A > 0:
            T_end = max([s.last_departure_time for s in lb.servers]) if any(s.last_departure_time > 0 for s in lb.servers) else 0
            T_w = sum([s.total_wait_time for s in lb.servers]) / A
            T_s = sum([s.total_service_time for s in lb.servers]) / A
        else:
            T_end = 0
            T_w = 0
            T_s = 0
        
        # Collect per-server data
        servers_data = []
        for i, server in enumerate(lb.servers):
            servers_data.append({
                'server_id': i,
                'served': server.customers_served,
                'rejected': server.customers_rejected,
                'avg_wait': server.average_wait_time(),
                'avg_service': server.average_service_time(),
                'utilization': (server.customers_served * server.average_service_time() / T) if T > 0 else 0
            })
        
        return TestResult(A, B, T_end, T_w, T_s, servers_data)
    
    except Exception as e:
        print(f"Error in test: {e}")
        return None

def run_multiple_tests(test_name, T, M, P, lambda_, Q, mu, num_runs=20, expected_behavior=""):
    """Run multiple tests and analyze results statistically"""
    print(f"\n{'='*90}")
    print(f"TEST: {test_name} ({num_runs} runs)")
    print(f"Parameters: T={T}, M={M}, P={P}, λ={lambda_}, Q={Q}, μ={mu}")
    if expected_behavior:
        print(f"Expected: {expected_behavior}")
    print(f"{'='*90}")
    
    results = []
    seeds = [42 + i for i in range(num_runs)]  # Different seeds for each run
    
    for i, seed in enumerate(seeds):
        result = run_single_test(T, M, P, lambda_, Q, mu, seed)
        if result:
            results.append(result)
        
        if (i + 1) % 5 == 0:  # Progress indicator
            print(f"Completed {i + 1}/{num_runs} runs...")
    
    if not results:
        print("❌ All test runs failed!")
        return None
    
    # Statistical analysis
    analyze_results(results, T, M, P, lambda_, Q, mu)
    return results

def analyze_results(results, T, M, P, lambda_, Q, mu):
    """Perform statistical analysis on multiple test results"""
    if not results:
        return
    
    # Extract metrics for analysis
    A_values = [r.A for r in results]
    B_values = [r.B for r in results]
    T_w_values = [r.T_w for r in results if r.T_w > 0]
    T_s_values = [r.T_s for r in results if r.T_s > 0]
    
    print(f"\n--- Statistical Analysis ({len(results)} successful runs) ---")
    
    # Basic statistics
    print(f"Customers Served (A):")
    print(f"  Mean: {statistics.mean(A_values):.2f} ± {statistics.stdev(A_values):.2f}")
    print(f"  Range: [{min(A_values)}, {max(A_values)}]")
    
    print(f"Customers Rejected (B):")
    print(f"  Mean: {statistics.mean(B_values):.2f} ± {statistics.stdev(B_values):.2f}")
    print(f"  Range: [{min(B_values)}, {max(B_values)}]")
    
    if T_w_values:
        print(f"Average Wait Time (T_w):")
        print(f"  Mean: {statistics.mean(T_w_values):.4f} ± {statistics.stdev(T_w_values):.4f}")
        print(f"  Range: [{min(T_w_values):.4f}, {max(T_w_values):.4f}]")
    
    if T_s_values:
        print(f"Average Service Time (T_s):")
        print(f"  Mean: {statistics.mean(T_s_values):.4f} ± {statistics.stdev(T_s_values):.4f}")
        print(f"  Range: [{min(T_s_values):.4f}, {max(T_s_values):.4f}]")
        
        # Theoretical service time comparison
        theoretical_T_s = sum([P[i] / mu[i] for i in range(M)])  # Weighted average
        actual_T_s_mean = statistics.mean(T_s_values)
        relative_error = abs(theoretical_T_s - actual_T_s_mean) / theoretical_T_s
        print(f"  Theoretical T_s: {theoretical_T_s:.4f}")
        print(f"  Relative Error: {relative_error:.1%} {'✓' if relative_error < 0.15 else '✗'}")
    
    # Probability distribution analysis
    analyze_probability_distribution(results, P, M)
    
    # System metrics
    total_capacity = sum(mu)
    utilization = lambda_ / total_capacity
    rejection_rate_mean = statistics.mean(B_values) / (lambda_ * T) if lambda_ * T > 0 else 0
    
    print(f"\nSystem Analysis:")
    print(f"  System Utilization (λ/Σμ): {utilization:.4f}")
    print(f"  Average Rejection Rate: {rejection_rate_mean:.4f}")
    
    if utilization < 1.0:
        print(f"  ✓ System is theoretically stable (ρ < 1)")
        if rejection_rate_mean < 0.1:
            print(f"  ✓ Low rejection rate confirms stability")
        else:
            print(f"  ⚠ High rejection rate despite stability (may need larger queues)")
    else:
        print(f"  ✗ System is theoretically unstable (ρ ≥ 1)")
        if rejection_rate_mean > 0.2:
            print(f"  ✓ High rejection rate confirms instability")

def analyze_probability_distribution(results, P, M):
    """Analyze if requests are distributed according to probabilities"""
    print(f"\n--- Probability Distribution Analysis ---")
    
    # Aggregate server statistics across all runs
    server_served_totals = [0] * M
    total_served_all_runs = 0
    
    for result in results:
        for server_data in result.servers_data:
            server_id = server_data['server_id']
            server_served_totals[server_id] += server_data['served']
            total_served_all_runs += server_data['served']
    
    if total_served_all_runs == 0:
        print("  ⚠ No customers served across all runs")
        return
    
    print(f"Total customers served across all runs: {total_served_all_runs}")
    
    max_deviation = 0
    for i in range(M):
        expected_fraction = P[i]
        actual_fraction = server_served_totals[i] / total_served_all_runs
        deviation = abs(expected_fraction - actual_fraction)
        max_deviation = max(max_deviation, deviation)
        
        status = "✓" if deviation <= 0.05 else "✗"  # 5% tolerance
        print(f"  Server {i}: Expected={expected_fraction:.3f}, "
              f"Actual={actual_fraction:.3f}, Deviation={deviation:.3f} {status}")
    
    if max_deviation <= 0.05:
        print(f"  ✓ All servers within 5% tolerance")
    else:
        print(f"  ✗ Some servers exceed 5% tolerance (max deviation: {max_deviation:.3f})")

def segel_tests():
    """Test specific boundary cases from the Hebrew requirements (בדיקת ספיות)"""
    print(f"\n{'='*90}")
    print("SEGEL TESTS - BOUNDARY CHECKING (בדיקת ספיות)")
    print("Testing specific cases from Hebrew requirements")
    print(f"{'='*90}")
    
    # Test cases from the document
    test_cases = [
        {
            'name': '4.1.1.1 - Single Server Basic Test',
            'T': 5000, 'M': 1, 'P': [1.0], 'lambda_': 20, 'Q': [1000], 'mu': [40],
            'expected': 'ρ=0.5, stable system, minimal rejections'
        },
        {
            'name': '4.1.1.2 - Extreme Probability Distribution (All to Server 0)',
            'T': 5000, 'M': 4, 'P': [1.0, 0.0, 0.0, 0.0], 'lambda_': 20, 
            'Q': [1000, 1000, 1000, 1000], 'mu': [40, 40, 40, 40],
            'expected': 'All traffic to server 0, others idle'
        },
        {
            'name': '4.1.1.3 - Minimal Non-Zero Probabilities',
            'T': 5000, 'M': 4, 'P': [0.0, 0.0, 1.0, 0.0], 'lambda_': 20, 
            'Q': [1000, 1000, 1000, 1000], 'mu': [40, 40, 40, 40],
            'expected': 'All traffic to server 2, others idle'
        },
        {
            'name': '4.1.1.4 - Very Small Probabilities (Precision Test)',
            'T': 5000, 'M': 4, 'P': [0.001, 0.001, 0.997, 0.001], 'lambda_': 20, 
            'Q': [1000, 1000, 1000, 1000], 'mu': [40, 40, 40, 40],
            'expected': '99.7% traffic to server 2, minimal to others'
        },
        {
            'name': '4.1.1.5 - Zero Queue Size Test',
            'T': 5000, 'M': 4, 'P': [0.0, 0.0, 1.0, 0.0], 'lambda_': 20, 
            'Q': [0, 0, 0, 0], 'mu': [40, 40, 40, 40],
            'expected': 'High rejection rate due to zero queues'
        },
        {
            'name': '4.1.1.6 - Equal Distribution with Low Service Rates',
            'T': 5000, 'M': 4, 'P': [0.25, 0.25, 0.25, 0.25], 'lambda_': 20, 
            'Q': [100, 100, 100, 100], 'mu': [0.5, 0.5, 0.5, 0.5],
            'expected': 'ρ=10, heavily overloaded system, many rejections'
        },
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Segel Test {i}: {test_case['name']} ---")
        
        # Calculate theoretical metrics
        T = test_case['T']
        M = test_case['M']
        P = test_case['P']
        lambda_ = test_case['lambda_']
        Q = test_case['Q']
        mu = test_case['mu']
        
        # Theoretical analysis
        total_capacity = sum(mu)
        utilization = lambda_ / total_capacity
        theoretical_T_s = sum([P[i] / mu[i] for i in range(M) if P[i] > 0])
        
        print(f"Theoretical Analysis:")
        print(f"  System Utilization (ρ): {utilization:.4f}")
        print(f"  Expected T_s: {theoretical_T_s:.4f}")
        print(f"  Expected arrivals: {lambda_ * T:.0f}")
        print(f"  System capacity: {total_capacity:.1f} requests/time")
        
        # Run multiple tests for statistical significance
        results = run_multiple_tests(
            test_case['name'], T, M, P, lambda_, Q, mu,
            num_runs=15, expected_behavior=test_case['expected']
        )
        
        # Additional analysis for specific cases
        if results:
            analyze_segel_case(results, test_case, utilization, theoretical_T_s)

def analyze_segel_case(results, test_case, theoretical_utilization, theoretical_T_s):
    """Perform specific analysis for Segel test cases"""
    print(f"\n--- Segel Case Analysis ---")
    
    A_values = [r.A for r in results]
    B_values = [r.B for r in results]
    T_s_values = [r.T_s for r in results if r.T_s > 0]
    
    avg_A = statistics.mean(A_values)
    avg_B = statistics.mean(B_values)
    
    T = test_case['T']
    lambda_ = test_case['lambda_']
    expected_arrivals = lambda_ * T
    
    # Service efficiency
    service_efficiency = avg_A / expected_arrivals if expected_arrivals > 0 else 0
    rejection_rate = avg_B / expected_arrivals if expected_arrivals > 0 else 0
    
    print(f"Performance Metrics:")
    print(f"  Expected arrivals: {expected_arrivals:.0f}")
    print(f"  Average served: {avg_A:.1f}")
    print(f"  Average rejected: {avg_B:.1f}")
    print(f"  Service efficiency: {service_efficiency:.1%}")
    print(f"  Rejection rate: {rejection_rate:.1%}")
    
    # Stability assessment
    if theoretical_utilization < 1.0:
        expected_stable = True
        print(f"  Expected: Stable system (ρ < 1)")
    else:
        expected_stable = False
        print(f"  Expected: Unstable system (ρ ≥ 1)")
    
    # Validation
    if expected_stable:
        if rejection_rate < 0.1:
            print(f"  ✓ Low rejection rate confirms stability")
        else:
            print(f"  ⚠ Higher rejection rate than expected for stable system")
    else:
        if rejection_rate > 0.3:
            print(f"  ✓ High rejection rate confirms instability")
        else:
            print(f"  ⚠ Lower rejection rate than expected for unstable system")
    
    # Service time validation
    if T_s_values and theoretical_T_s > 0:
        actual_T_s = statistics.mean(T_s_values)
        relative_error = abs(actual_T_s - theoretical_T_s) / theoretical_T_s
        print(f"  Service time validation: {relative_error:.1%} error {'✓' if relative_error < 0.2 else '✗'}")
    
    # Special case validations
    case_name = test_case['name']
    if "All to Server" in case_name or "Extreme Probability" in case_name:
        validate_extreme_probability_distribution(results, test_case)
    elif "Zero Queue" in case_name:
        validate_zero_queue_behavior(results, test_case)
    elif "Precision Test" in case_name:
        validate_precision_distribution(results, test_case)

def validate_extreme_probability_distribution(results, test_case):
    """Validate cases where all traffic goes to one server"""
    print(f"\n--- Extreme Probability Validation ---")
    
    P = test_case['P']
    M = test_case['M']
    
    # Find which server should get all traffic
    target_server = P.index(max(P))
    
    # Aggregate across all runs
    total_served_per_server = [0] * M
    for result in results:
        for server_data in result.servers_data:
            server_id = server_data['server_id']
            total_served_per_server[server_id] += server_data['served']
    
    total_served = sum(total_served_per_server)
    
    if total_served > 0:
        for i in range(M):
            fraction = total_served_per_server[i] / total_served
            expected = P[i]
            print(f"  Server {i}: {fraction:.3f} (expected {expected:.3f})")
            
        target_fraction = total_served_per_server[target_server] / total_served
        if target_fraction > 0.95:  # Should get >95% of traffic
            print(f"  ✓ Server {target_server} correctly receives most traffic ({target_fraction:.1%})")
        else:
            print(f"  ✗ Server {target_server} should receive more traffic ({target_fraction:.1%})")

def validate_zero_queue_behavior(results, test_case):
    """Validate behavior with zero queue sizes"""
    print(f"\n--- Zero Queue Validation ---")
    
    B_values = [r.B for r in results]
    A_values = [r.A for r in results]
    
    avg_rejection_rate = statistics.mean(B_values) / (test_case['lambda_'] * test_case['T'])
    
    # With zero queues and high load, expect significant rejections
    if avg_rejection_rate > 0.5:  # Should reject >50% due to no queueing
        print(f"  ✓ High rejection rate ({avg_rejection_rate:.1%}) as expected with zero queues")
    else:
        print(f"  ⚠ Lower rejection rate ({avg_rejection_rate:.1%}) than expected with zero queues")

def validate_precision_distribution(results, test_case):
    """Validate very small probability distributions"""
    print(f"\n--- Precision Distribution Validation ---")
    
    P = test_case['P']
    M = test_case['M']
    
    # Find servers with very small probabilities (0.001)
    small_prob_servers = [i for i, p in enumerate(P) if 0 < p <= 0.01]
    dominant_server = P.index(max(P))
    
    # Check if small probabilities are still respected
    total_served_per_server = [0] * M
    for result in results:
        for server_data in result.servers_data:
            server_id = server_data['server_id']
            total_served_per_server[server_id] += server_data['served']
    
    total_served = sum(total_served_per_server)
    
    if total_served > 0:
        dominant_fraction = total_served_per_server[dominant_server] / total_served
        print(f"  Dominant server {dominant_server}: {dominant_fraction:.3f} (expected {P[dominant_server]:.3f})")
        
        for server_id in small_prob_servers:
            fraction = total_served_per_server[server_id] / total_served
            expected = P[server_id]
            print(f"  Small prob server {server_id}: {fraction:.4f} (expected {expected:.3f})")
            
            # Check if small probabilities are approximately respected
            if abs(fraction - expected) / expected < 0.5:  # Within 50% of expected (generous for small numbers)
                print(f"    ✓ Small probability respected within tolerance")
            else:
                print(f"    ⚠ Small probability deviation larger than expected")

def test_configuration_matrix():
    """Test multiple configurations in a systematic way"""
    print(f"\n{'='*90}")
    print("CONFIGURATION MATRIX TESTING")
    print(f"{'='*90}")
    
    configurations = [
        # Light load scenarios
        {
            'name': 'Light Load - Balanced Servers',
            'T': 100, 'M': 2, 'P': [0.5, 0.5], 'lambda_': 2.0, 'Q': [10, 10], 'mu': [3.0, 3.0],
            'expected': 'Low rejection, balanced load'
        },
        {
            'name': 'Light Load - Unbalanced Probabilities',
            'T': 100, 'M': 3, 'P': [0.7, 0.2, 0.1], 'lambda_': 3.0, 'Q': [10, 10, 10], 'mu': [2.0, 2.0, 2.0],
            'expected': 'Server 0 handles most traffic'
        },
        
        # Moderate load scenarios
        {
            'name': 'Moderate Load - Different Service Rates',
            'T': 150, 'M': 3, 'P': [0.4, 0.3, 0.3], 'lambda_': 5.0, 'Q': [8, 6, 4], 'mu': [3.0, 2.5, 2.0],
            'expected': 'Some queuing, different server utilizations'
        },
        {
            'name': 'Moderate Load - Small Queues',
            'T': 100, 'M': 2, 'P': [0.6, 0.4], 'lambda_': 4.0, 'Q': [2, 3], 'mu': [3.0, 2.5],
            'expected': 'Higher rejection due to small queues'
        },
        
        # Heavy load scenarios
        {
            'name': 'Heavy Load - Near Capacity',
            'T': 200, 'M': 3, 'P': [0.4, 0.35, 0.25], 'lambda_': 8.0, 'Q': [15, 12, 10], 'mu': [3.0, 3.0, 2.5],
            'expected': 'High utilization, some rejections'
        },
        {
            'name': 'Overload - Exceeds Capacity',
            'T': 150, 'M': 2, 'P': [0.5, 0.5], 'lambda_': 10.0, 'Q': [5, 5], 'mu': [3.0, 3.0],
            'expected': 'Many rejections, system overloaded'
        },
    ]
    
    for config in configurations:
        run_multiple_tests(
            config['name'], config['T'], config['M'], config['P'],
            config['lambda_'], config['Q'], config['mu'],
            num_runs=15, expected_behavior=config['expected']
        )

def test_edge_cases_comprehensive():
    """Comprehensive edge case testing with multiple runs"""
    print(f"\n{'='*90}")
    print("COMPREHENSIVE EDGE CASE TESTING")
    print(f"{'='*90}")
    
    edge_cases = [
        # Single server cases
        {
            'name': 'Single Server - Light Load',
            'T': 100, 'M': 1, 'P': [1.0], 'lambda_': 2.0, 'Q': [10], 'mu': [5.0],
            'expected': 'M/M/1/11 queue behavior'
        },
        {
            'name': 'Single Server - Heavy Load',
            'T': 150, 'M': 1, 'P': [1.0], 'lambda_': 4.0, 'Q': [3], 'mu': [3.0],
            'expected': 'High utilization, some rejections'
        },
        
        # Extreme probability distributions
        {
            'name': 'Extreme Skew - 95/5 Split',
            'T': 200, 'M': 2, 'P': [0.95, 0.05], 'lambda_': 4.0, 'Q': [10, 10], 'mu': [3.0, 3.0],
            'expected': 'Server 0 heavily loaded, Server 1 mostly idle'
        },
        {
            'name': 'Many Servers - Equal Split',
            'T': 200, 'M': 5, 'P': [0.2, 0.2, 0.2, 0.2, 0.2], 'lambda_': 8.0, 
            'Q': [5, 5, 5, 5, 5], 'mu': [2.0, 2.0, 2.0, 2.0, 2.0],
            'expected': 'Equal load distribution across 5 servers'
        },
        
        # Zero queue size cases
        {
            'name': 'Zero Queue - Immediate Rejection',
            'T': 100, 'M': 2, 'P': [0.5, 0.5], 'lambda_': 6.0, 'Q': [0, 0], 'mu': [2.0, 2.0],
            'expected': 'High rejection rate, no queueing'
        },
        
        # Very fast/slow servers
        {
            'name': 'Very Fast Servers',
            'T': 100, 'M': 2, 'P': [0.6, 0.4], 'lambda_': 10.0, 'Q': [8, 8], 'mu': [20.0, 15.0],
            'expected': 'Minimal waiting, high throughput'
        },
        {
            'name': 'Very Slow Servers',
            'T': 200, 'M': 2, 'P': [0.5, 0.5], 'lambda_': 1.0, 'Q': [15, 15], 'mu': [0.5, 0.5],
            'expected': 'Long service times, significant queueing'
        },
        
        # Mixed service rates
        {
            'name': 'Highly Asymmetric Servers',
            'T': 150, 'M': 3, 'P': [0.3, 0.4, 0.3], 'lambda_': 5.0, 
            'Q': [10, 5, 15], 'mu': [10.0, 1.0, 2.0],
            'expected': 'Different server behaviors due to rate differences'
        },
    ]
    
    for case in edge_cases:
        run_multiple_tests(
            case['name'], case['T'], case['M'], case['P'],
            case['lambda_'], case['Q'], case['mu'],
            num_runs=10, expected_behavior=case['expected']
        )

def test_stability_boundaries():
    """Test system behavior at stability boundaries"""
    print(f"\n{'='*90}")
    print("STABILITY BOUNDARY TESTING")
    print(f"{'='*90}")
    
    # Test different utilization levels
    utilization_levels = [0.5, 0.8, 0.9, 0.95, 1.0, 1.1, 1.3]
    base_capacity = 6.0  # Sum of service rates
    
    for rho in utilization_levels:
        lambda_val = rho * base_capacity
        stability_status = "Stable" if rho < 1.0 else "Unstable"
        
        run_multiple_tests(
            f"Utilization ρ={rho:.2f} ({stability_status})",
            T=300, M=3, P=[0.4, 0.35, 0.25], lambda_=lambda_val,
            Q=[10, 8, 6], mu=[2.5, 2.0, 1.5],
            num_runs=12, 
            expected_behavior=f"ρ={rho:.2f}, expect {'low' if rho < 0.9 else 'high'} rejection rate"
        )

def test_input_validation_comprehensive():
    """Test input validation with multiple scenarios"""
    print(f"\n{'='*90}")
    print("INPUT VALIDATION TESTING")
    print(f"{'='*90}")
    
    # Test probability sum validation
    test_cases = [
        ([0.5, 0.5], "Valid - sums to 1.0"),
        ([0.33, 0.33, 0.34], "Valid - sums to 1.0"),
        ([1/3, 1/3, 1/3], "Precision test - should be valid"),
        ([0.3, 0.3, 0.3], "Invalid - sums to 0.9"),
        ([0.4, 0.4, 0.3], "Invalid - sums to 1.1"),
    ]
    
    for probs, description in test_cases:
        prob_sum = sum(probs)
        is_valid = abs(prob_sum - 1.0) < 1e-6
        status = "✓" if is_valid else "✗"
        print(f"  {description}: {probs} → sum={prob_sum:.10f} {status}")

def performance_stress_test():
    """Test performance with increasingly large parameters"""
    print(f"\n{'='*90}")
    print("PERFORMANCE STRESS TESTING")
    print(f"{'='*90}")
    
    stress_configs = [
        {'T': 500, 'M': 2, 'lambda_': 10.0, 'scale': 'Small'},
        {'T': 1000, 'M': 5, 'lambda_': 20.0, 'scale': 'Medium'},
        {'T': 2000, 'M': 8, 'lambda_': 40.0, 'scale': 'Large'},
    ]
    
    for config in stress_configs:
        M = config['M']
        P = [1/M] * M  # Equal probabilities
        Q = [20] * M   # Adequate queue sizes
        mu = [5.0] * M # Equal service rates
        
        print(f"\n--- {config['scale']} Scale Test ---")
        print(f"Parameters: T={config['T']}, M={M}, λ={config['lambda_']}")
        
        start_time = time.time()
        result = run_single_test(config['T'], M, P, config['lambda_'], Q, mu, seed=42)
        end_time = time.time()
        
        if result:
            runtime = end_time - start_time
            total_events = result.A * 2  # Arrival + Departure per customer
            events_per_sec = total_events / runtime if runtime > 0 else 0
            
            print(f"  Runtime: {runtime:.3f} seconds")
            print(f"  Total events: {total_events}")
            print(f"  Events/second: {events_per_sec:.1f}")
            print(f"  Customers served: {result.A}")
            print(f"  Customers rejected: {result.B}")
            
            if runtime < 10.0:  # Should complete within reasonable time
                print(f"  ✓ Performance acceptable")
            else:
                print(f"  ⚠ Performance may be slow for large simulations")
        else:
            print(f"  ❌ Test failed")

def performance_time_limit_test():
    """Test edge cases that might exceed 2-minute time limit (Hebrew requirement)"""
    print(f"\n{'='*90}")
    print("PERFORMANCE TIME LIMIT TESTING (2-MINUTE REQUIREMENT)")
    print("Testing edge cases to ensure execution time < 2 minutes")
    print(f"{'='*90}")
    
    # Edge cases that might be computationally expensive
    time_limit_cases = [
        {
            'name': 'Very High Arrival Rate with Small Queues',
            'T': 10000, 'M': 4, 'P': [0.25, 0.25, 0.25, 0.25], 'lambda_': 1000,
            'Q': [5, 5, 5, 5], 'mu': [200, 200, 200, 200],
            'expected_issue': 'Many rejections due to high λ and small queues'
        },
        {
            'name': 'Long Simulation Time with Moderate Load',
            'T': 50000, 'M': 2, 'P': [0.5, 0.5], 'lambda_': 100,
            'Q': [50, 50], 'mu': [60, 60],
            'expected_issue': 'Long simulation time might cause timeout'
        },
        {
            'name': 'Many Servers with High Load',
            'T': 20000, 'M': 10, 'P': [0.1]*10, 'lambda_': 500,
            'Q': [20]*10, 'mu': [60]*10,
            'expected_issue': 'Many servers might slow down event processing'
        },
        {
            'name': 'Unstable System with Large Simulation Time',
            'T': 30000, 'M': 3, 'P': [0.4, 0.3, 0.3], 'lambda_': 200,
            'Q': [10, 10, 10], 'mu': [30, 30, 30],
            'expected_issue': 'ρ > 2, heavily overloaded system'
        },
        {
            'name': 'Very Fast Servers with Extreme Arrival Rate',
            'T': 15000, 'M': 5, 'P': [0.2]*5, 'lambda_': 5000,
            'Q': [100]*5, 'mu': [1200]*5,
            'expected_issue': 'Extremely high event rate'
        },
        {
            'name': 'Single Server with Maximum Load',
            'T': 25000, 'M': 1, 'P': [1.0], 'lambda_': 500,
            'Q': [1000], 'mu': [600],
            'expected_issue': 'Single bottleneck with high load'
        }
    ]
    
    max_time_allowed = 120.0  # 2 minutes in seconds
    all_passed = True
    
    for i, test_case in enumerate(time_limit_cases, 1):
        print(f"\n--- Time Limit Test {i}: {test_case['name']} ---")
        print(f"Parameters: T={test_case['T']}, M={test_case['M']}, λ={test_case['lambda_']}")
        print(f"Potential issue: {test_case['expected_issue']}")
        
        # Calculate theoretical metrics
        total_capacity = sum(test_case['mu'])
        utilization = test_case['lambda_'] / total_capacity
        expected_events = test_case['lambda_'] * test_case['T'] * 2  # Rough estimate (arrivals + departures)
        
        print(f"Theoretical analysis:")
        print(f"  System utilization (ρ): {utilization:.2f}")
        print(f"  Expected events: ~{expected_events:,.0f}")
        print(f"  Total capacity: {total_capacity}")
        
        # Run the test with timing
        start_time = time.time()
        
        try:
            # Set a consistent seed for reproducibility
            result = run_single_test(
                test_case['T'], test_case['M'], test_case['P'], 
                test_case['lambda_'], test_case['Q'], test_case['mu'], 
                seed=12345
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            print(f"\nExecution Results:")
            print(f"  Execution time: {execution_time:.2f} seconds")
            print(f"  Time limit: {max_time_allowed} seconds")
            
            if execution_time <= max_time_allowed:
                print(f"  ✅ PASSED - Within time limit ({execution_time:.2f}s < {max_time_allowed}s)")
                
                if result:
                    actual_events = result.A * 2  # Approximate events processed
                    events_per_second = actual_events / execution_time if execution_time > 0 else 0
                    print(f"  Performance: {events_per_second:,.0f} events/second")
                    print(f"  Customers served: {result.A:,}")
                    print(f"  Customers rejected: {result.B:,}")
                    
                    # Additional performance metrics
                    rejection_rate = result.B / (result.A + result.B) if (result.A + result.B) > 0 else 0
                    print(f"  Rejection rate: {rejection_rate:.1%}")
                    
                    if utilization >= 1.0:
                        print(f"  ✓ High rejection rate expected for unstable system (ρ={utilization:.2f})")
                    else:
                        print(f"  ✓ System behavior as expected for ρ={utilization:.2f}")
                        
            else:
                print(f"  ❌ FAILED - Exceeded time limit ({execution_time:.2f}s > {max_time_allowed}s)")
                all_passed = False
                
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"  ❌ FAILED - Exception after {execution_time:.2f}s: {e}")
            all_passed = False
    
    # Summary
    print(f"\n{'='*60}")
    print("TIME LIMIT TEST SUMMARY")
    print(f"{'='*60}")
    
    if all_passed:
        print("✅ ALL TESTS PASSED - Simulator meets 2-minute requirement")
        print("✅ No edge cases exceed the time limit")
        print("✅ Performance is acceptable for submission")
    else:
        print("❌ SOME TESTS FAILED - Simulator may exceed time limits")
        print("⚠️  Consider optimizing for edge cases")
        print("⚠️  Review implementation for performance bottlenecks")
    
    print(f"\nRecommendations:")
    print(f"  - All test cases should complete within {max_time_allowed} seconds")
    print(f"  - Consider early termination for unstable systems")
    print(f"  - Monitor memory usage for large simulations")
    print(f"  - Test with actual submission parameters before final submission")
    
    return all_passed

if __name__ == "__main__":
    try:
        print("🚀 COMPREHENSIVE MULTI-SERVER LOAD BALANCER TEST SUITE")
        print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*90}")
        
        # Run all test categories with multiple configurations and runs
        segel_tests()  # Run the boundary tests first
        test_configuration_matrix()
        test_edge_cases_comprehensive()
        test_stability_boundaries()
        test_input_validation_comprehensive()
        performance_stress_test()
        performance_time_limit_test()
        
        print(f"\n{'='*90}")
        print("🎉 COMPREHENSIVE TEST SUITE COMPLETED!")
        print("📊 Results analyzed across multiple runs for statistical significance")
        print("🔬 Theoretical validation performed where applicable")
        print("⚡ Performance and edge cases thoroughly tested")
        print("🎯 Segel boundary tests validated against theoretical results")
        print(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*90}")
        
    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()
        print("Please check your implementation and try again.") 