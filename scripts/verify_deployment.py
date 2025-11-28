"""
Verification script for Hybrid ML Scheduler deployment.
Checks all API endpoints and verifies system health.
"""
import requests
import sys
import time
import json
from termcolor import colored

BASE_URL = "http://localhost:8000"

def print_status(message, status="INFO"):
    if status == "INFO":
        print(colored(f"[INFO] {message}", "blue"))
    elif status == "SUCCESS":
        print(colored(f"[SUCCESS] {message}", "green"))
    elif status == "ERROR":
        print(colored(f"[ERROR] {message}", "red"))
    elif status == "WARNING":
        print(colored(f"[WARNING] {message}", "yellow"))

def check_endpoint(path, method="GET", expected_status=200, description=""):
    url = f"{BASE_URL}{path}"
    try:
        start_time = time.time()
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url)
        duration = (time.time() - start_time) * 1000
        
        if response.status_code == expected_status:
            print_status(f"✓ {method} {path} - {response.status_code} ({duration:.2f}ms) - {description}", "SUCCESS")
            return True, response
        else:
            print_status(f"✗ {method} {path} - Expected {expected_status}, got {response.status_code}", "ERROR")
            print(f"Response: {response.text}")
            return False, response
    except Exception as e:
        print_status(f"✗ {method} {path} - Connection failed: {e}", "ERROR")
        return False, None

def verify_deployment():
    print_status("Starting deployment verification...", "INFO")
    
    # Wait for server to be ready
    max_retries = 5
    for i in range(max_retries):
        try:
            requests.get(f"{BASE_URL}/health")
            break
        except requests.ConnectionError:
            if i == max_retries - 1:
                print_status("Server is not reachable after 5 retries. Is it running?", "ERROR")
                sys.exit(1)
            print_status(f"Waiting for server... ({i+1}/{max_retries})", "WARNING")
            time.sleep(2)

    success_count = 0
    total_checks = 0

    # 1. Health Checks
    print("\n--- Health Checks ---")
    checks = [
        ("/health", "GET", 200, "Basic health check"),
        ("/health/ready", "GET", 200, "Readiness check (DB+Redis)"),
        ("/health/live", "GET", 200, "Liveness check"),
        ("/health/info", "GET", 200, "System info & version"),
    ]
    
    for path, method, status, desc in checks:
        total_checks += 1
        ok, _ = check_endpoint(path, method, status, desc)
        if ok: success_count += 1

    # 2. Metrics & Observability
    print("\n--- Metrics & Observability ---")
    checks = [
        ("/metrics/prometheus", "GET", 200, "Prometheus metrics"),
        ("/metrics/cache", "GET", 200, "Cache statistics"),
        ("/observability/health-check", "GET", 200, "Observability health"),
        ("/observability/logs/tail?lines=5", "GET", 200, "Log viewing"),
    ]
    
    for path, method, status, desc in checks:
        total_checks += 1
        ok, _ = check_endpoint(path, method, status, desc)
        if ok: success_count += 1

    # 3. Simulation Control
    print("\n--- Simulation Control ---")
    # Check status first
    total_checks += 1
    ok, resp = check_endpoint("/api/status", "GET", 200, "Simulation status")
    if ok:
        success_count += 1
        data = resp.json()
        print(f"   Status: Running={data.get('is_running')}, Paused={data.get('is_paused')}")
        print(f"   Tasks Processed: {data.get('tasks_processed')}")

    # 4. Security Headers
    print("\n--- Security Headers ---")
    total_checks += 1
    ok, resp = check_endpoint("/health", "GET", 200, "Checking security headers")
    if ok:
        headers = resp.headers
        security_headers = [
            "Strict-Transport-Security",
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Content-Security-Policy"
        ]
        
        all_headers_present = True
        for h in security_headers:
            if h in headers:
                print_status(f"✓ Header present: {h}", "SUCCESS")
            else:
                print_status(f"✗ Missing header: {h}", "ERROR")
                all_headers_present = False
        
        if all_headers_present:
            success_count += 1

    # 5. Rate Limiting
    print("\n--- Rate Limiting ---")
    print_status("Testing rate limiting (sending 5 requests)...", "INFO")
    rate_limit_ok = True
    for i in range(5):
        ok, resp = check_endpoint("/health", "GET", 200, f"Request {i+1}")
        if not ok: rate_limit_ok = False
        if "X-RateLimit-Remaining" in resp.headers:
            print(f"   Remaining: {resp.headers['X-RateLimit-Remaining']}")
    
    if rate_limit_ok:
        success_count += 1
        total_checks += 1

    # Summary
    print("\n--- Verification Summary ---")
    if success_count == total_checks:
        print_status(f"ALL CHECKS PASSED ({success_count}/{total_checks})", "SUCCESS")
        sys.exit(0)
    else:
        print_status(f"SOME CHECKS FAILED ({success_count}/{total_checks})", "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    verify_deployment()
