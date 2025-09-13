import random
import time

def get_current_wait_time(hospital: str) -> int | str:
    """Dummy function to generate fake wait times for a hospital."""
    
    valid_hospitals = ["A", "B", "C", "D"]

    if hospital not in valid_hospitals:
        return f"Hospital {hospital} does not exist"

    # Simulate an API delay (e.g., network latency)
    time.sleep(1)

    # Return a random wait time in minutes
    return random.randint(0, 10000)
