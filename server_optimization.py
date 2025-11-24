import csv
import numpy as np


def read_observations(path):
    arrivals, services = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            arrivals.append(float(row["arrival"]))
            services.append(float(row["service"]))
    return arrivals, services


def estimate_rates(arrivals, services):
    arrivals = sorted(arrivals)                         # Ensure arrivals are in order
    inter = np.diff([0] + arrivals)                     # Inter-arrival times (prepend 0 for first arrival)
    return 1 / np.mean(inter), 1 / np.mean(services)    # Return lambda and mu


def generate_synthetic(n, lam, mu, seed=123):
    rng = np.random.default_rng()       # Create random number generator (I removed seed for variability)
    inter = rng.exponential(1 / lam, n) # Simulated inter-arrival times
    svc = rng.exponential(1 / mu, n)    # Simulated service times
    arr = np.cumsum(inter)              # Simulated arrival times
    return arr, svc                     # Return arrival and service times


def simulate_fcfs(arrival, service, servers):
    next_free = np.zeros(servers)   # Next free time for each server (first all 0)
    busy = np.zeros(servers)        # Total busy time for each server (first all 0)
    waits = []                      # List of wait times
    system_times = []               # List of system times

    for a, s in zip(arrival, service):              # For each arrival and service time
        free_server = np.argmin(next_free)          # Find the next free server
        start = max(a, next_free[free_server])      # Start time is max of arrival and server free time
        depart = start + s                          # Departure time

        waits.append(start - a)                     # Record wait time
        system_times.append(depart - a)             # Record system time

        next_free[free_server] = depart             # Update server next free time
        busy[free_server] += s                      # Update server busy time

    sim_time = next_free.max() - arrival[0]         # Total simulation time
    lam_hat = len(arrival) / sim_time               # Estimated arrival rate

    return {
        "avg_wait": np.mean(waits),
        "avg_system": np.mean(system_times),
        "Lq": lam_hat * np.mean(waits),
        "L": lam_hat * np.mean(system_times),
        "utilization": busy.sum() / (servers * sim_time),
        "sim_time": sim_time,
        "lambda_hat": lam_hat,
    }


def find_best_server_count(arrival, service, max_servers=10, alpha=1.0, beta=0.5):
    """Return best # of servers based on tradeoff score, and print all results."""
    best = None

    print("=== Server-by-server performance ===")
    print(f"{'S':>2} | {'Wq':>8} | {'W':>8} | {'Lq':>8} | {'L':>8} | {'Util':>6} | {'Score':>10}")
    print("-" * 70)

    for s in range(1, max_servers + 1):         # For each server count
        r = simulate_fcfs(arrival, service, s)  # Simulate FCFS with s servers

        score = alpha * r["avg_wait"] + beta * (1 - r["utilization"]) # Compute score (balance wait and utilization)

        print(f"{s:>2} | "
              f"{r['avg_wait']:8.3f} | "
              f"{r['avg_system']:8.3f} | "
              f"{r['Lq']:8.3f} | "
              f"{r['L']:8.3f} | "
              f"{r['utilization']:6.3f} | "
              f"{score:10.4f}")

        if best is None or score < best["score"]: # Update best if first or better score
            best = {
                "servers": s,
                "score": score,
                "result": r,
            }

    print() 
    return best # Return best configuration



def main():
    arrivals_obs, services_obs = read_observations("observations.csv")  # Read data
    lam, mu = estimate_rates(arrivals_obs, services_obs)                # Estimate rates

    arrival, service = generate_synthetic(500, lam, mu)                 # Generate synthetic data

    best = find_best_server_count(arrival, service,
                                  max_servers=10,
                                  alpha=1.0,
                                  beta=0.5)                             # Find best server count  

    print("Best configuration found:")
    print(f"  Servers: {best['servers']}")
    print(f"  Score:   {best['score']:.4f}")
    r = best["result"]
    print()
    print("With performance:")
    print(f"  Avg wait: {r['avg_wait']:.3f}")
    print(f"  Util:     {r['utilization']:.3f}")
    print(f"  Lq:       {r['Lq']:.3f}")


if __name__ == "__main__":
    main()
