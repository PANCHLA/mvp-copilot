import requests
import time
import sys

def wait_for_server(url, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                print("[Test] Server is up!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
        print("[Test] Waiting for server...")
    return False

def test_query():
    # Change from localhost to 127.0.0.1 to match your server
    base_url = "http://127.0.0.1:8000"  # Changed from localhost to 127.0.0.1
    if not wait_for_server(f"{base_url}/health"):
        print("[Test] Server failed to start.")
        sys.exit(1)
    
    # Rest of the function remains the same

    print("[Test] Sending query...")
    payload = {"q": "explain auth", "k": 3, "mode": "stub"}
    try:
        r = requests.post(f"{base_url}/query", json=payload)
        r.raise_for_status()
        data = r.json()
        print("[Test] Response received:")
        print(data)
        if "answer" in data:
            print("[Test] SUCCESS: Answer field present.")
        else:
            print("[Test] FAILURE: Answer field missing.")
            sys.exit(1)
    except Exception as e:
        print(f"[Test] Query failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_query()