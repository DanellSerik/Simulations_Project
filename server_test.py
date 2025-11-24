import makerspace_sim as mss

def find_best_server_count(arrival, service, max_servers=10, alpha=1.0, beta=0.5):
    """Return best # of servers based on tradeoff score, and print all results."""
    best = None
    for s in range(1, max_servers + 1):         # For each server count
        r = mss.simulate_fcfs(arrival, service, s)  # Simulate FCFS with s servers

        score = alpha * r["avg_wait"] + beta * (1 - r["utilization"]) # Compute score (balance wait and utilization)

        if best is None or score < best["score"]: # Update best if first or better score
            best = {
                "servers": s,
                "score": score,
                "result": r,
            }

    return best # Return best configuration

def main():
    arrivals_obs, services_obs = mss.read_observations("observations.csv")  # Read data

    best = {"servers": 0}

    for i in range(100):
        lam, mu = mss.estimate_rates(arrivals_obs, services_obs)                # Estimate rates

        arrival, service = mss.generate_synthetic(500, lam, mu)                 # Generate synthetic data
        attempt = find_best_server_count(arrival, service,
                                  max_servers=10,
                                  alpha=1.0,
                                  beta=0.5)                             # Find best server count  
        best["servers"] += attempt["servers"]
    print("Best configuration found after 100 runs:")
    print(f"  Servers: {best['servers']/100}")

if __name__ == "__main__":
    main()
