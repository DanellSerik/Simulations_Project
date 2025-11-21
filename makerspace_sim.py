#!/usr/bin/env python3
"""
Multi-server queue simulator using observed data.

Reads arrival/service times, fits exponential distributions,
generates synthetic jobs, and simulates multi-server FCFS queue.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np


def read_observation_csv(path: str) -> Tuple[List[float], List[float]]:
    """Read observed arrivals and service times from CSV (arrival,service)."""
    arrivals: List[float] = []
    services: List[float] = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if "arrival" not in reader.fieldnames or "service" not in reader.fieldnames:
            raise ValueError("CSV must have columns: arrival,service")

        for row in reader:
            arrivals.append(float(row["arrival"]))
            services.append(float(row["service"]))

    if len(arrivals) == 0:
        raise ValueError("No data found in CSV")

    return arrivals, services


def estimate_exponential_params(arrivals: List[float],
                                services: List[float]) -> Tuple[float, float]:
    """Fit exponential distributions. Returns (lambda_hat, mu_hat)."""
    arrivals = sorted(arrivals)
    interarrivals = [arrivals[0]] + [
        arrivals[i] - arrivals[i - 1] for i in range(1, len(arrivals))
    ]

    mean_iat = sum(interarrivals) / len(interarrivals)
    mean_service = sum(services) / len(services)

    if mean_iat <= 0 or mean_service <= 0:
        raise ValueError("Mean interarrival and service times must be positive")

    lambda_hat = 1.0 / mean_iat
    mu_hat = 1.0 / mean_service
    return lambda_hat, mu_hat


def generate_synthetic_sample(
    num_customers: int,
    lambda_hat: float,
    mu_hat: float,
    seed: Optional[int] = None,
) -> Tuple[List[float], List[float]]:
    """Generate synthetic arrival and service times. Returns (arrival_times, service_times)."""
    rng = np.random.default_rng(seed)

    mean_iat = 1.0 / lambda_hat
    mean_service = 1.0 / mu_hat

    interarrivals = rng.exponential(scale=mean_iat, size=num_customers)
    services = rng.exponential(scale=mean_service, size=num_customers)

    arrival_times = np.cumsum(interarrivals)

    return arrival_times.tolist(), services.tolist()


@dataclass
class SimulationResult:
    num_servers: int
    num_customers: int
    sim_time: float
    lambda_hat: float
    avg_wait: float
    avg_system_time: float
    Lq_hat: float
    L_hat: float
    utilization: float


def simulate_multiserver_fcfs(
    arrival_times: List[float],
    service_times: List[float],
    num_servers: int,
) -> SimulationResult:
    """Simulate FCFS multi-server queue. Assigns each arrival to earliest-available server."""
    if num_servers <= 0:
        raise ValueError("num_servers must be positive")

    n = len(arrival_times)
    if n != len(service_times):
        raise ValueError("arrival_times and service_times must have same length")

    arrivals = list(arrival_times)
    services = list(service_times)

    next_free = [0.0] * num_servers
    busy_time = [0.0] * num_servers

    wait_times: List[float] = []
    system_times: List[float] = []

    for a, s in zip(arrivals, services):
        server_index = min(range(num_servers), key=lambda j: next_free[j])

        start_service = max(a, next_free[server_index])
        wait = start_service - a
        depart = start_service + s

        next_free[server_index] = depart
        busy_time[server_index] += s

        wait_times.append(wait)
        system_times.append(depart - a)

    sim_start = arrivals[0]
    sim_end = max(next_free)
    sim_time = sim_end - sim_start

    total_customers = n
    avg_wait = sum(wait_times) / total_customers
    avg_sys = sum(system_times) / total_customers

    lambda_hat = total_customers / sim_time if sim_time > 0 else float("nan")

    Lq_hat = lambda_hat * avg_wait
    L_hat = lambda_hat * avg_sys

    total_busy = sum(busy_time)
    utilization = total_busy / (num_servers * sim_time)

    return SimulationResult(
        num_servers=num_servers,
        num_customers=total_customers,
        sim_time=sim_time,
        lambda_hat=lambda_hat,
        avg_wait=avg_wait,
        avg_system_time=avg_sys,
        Lq_hat=Lq_hat,
        L_hat=L_hat,
        utilization=utilization,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Multi-printer queue simulation from observed data."
    )
    parser.add_argument(
        "--obs-csv",
        default="observations.csv",
        help="CSV file with observed arrival and service times.",
    )
    parser.add_argument(
        "--servers",
        type=int,
        default=3,
        help="Number of servers.",
    )
    parser.add_argument(
        "--num-customers",
        type=int,
        default=500,
        help="Number of synthetic customers to simulate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed.",
    )

    args = parser.parse_args()

    arrivals_obs, services_obs = read_observation_csv(args.obs_csv)
    lambda_hat, mu_hat = estimate_exponential_params(arrivals_obs, services_obs)

    print("=== Fitted exponential parameters from observations ===")
    print(f"  Mean interarrival time  = {1.0 / lambda_hat:.3f}")
    print(f"  Mean service time       = {1.0 / mu_hat:.3f}")
    print(f"  Arrival rate λ_hat      = {lambda_hat:.4f}")
    print(f"  Service rate μ_hat      = {mu_hat:.4f}")
    print()

    arrivals_sim, services_sim = generate_synthetic_sample(
        num_customers=args.num_customers,
        lambda_hat=lambda_hat,
        mu_hat=mu_hat,
        seed=args.seed,
    )

    result = simulate_multiserver_fcfs(
        arrival_times=arrivals_sim,
        service_times=services_sim,
        num_servers=args.servers,
    )

    print("=== Simulation results (synthetic multi-printer system) ===")
    print(f"  Number of printers (servers): {result.num_servers}")
    print(f"  Number of customers/jobs    : {result.num_customers}")
    print(f"  Simulated time horizon      : {result.sim_time:.3f}")
    print()
    print(f"  Average waiting time Wq_hat : {result.avg_wait:.3f}")
    print(f"  Average system time W_hat   : {result.avg_system_time:.3f}")
    print(f"  Estimated arrival rate λ    : {result.lambda_hat:.4f}")
    print(f"  Lq_hat (avg # in queue)     : {result.Lq_hat:.3f}")
    print(f"  L_hat  (avg # in system)    : {result.L_hat:.3f}")
    print(f"  Server utilization ρ_hat    : {result.utilization:.3f}")


if __name__ == "__main__":
    main()
