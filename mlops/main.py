import os
from datetime import datetime
import uuid

result_path = os.path.join("results", "results.txt")
os.makedirs(os.path.dirname(result_path), exist_ok=True)

with open(result_path, 'w') as f:
    f.write(f"{uuid.uuid4()} created at this point of time {datetime.now()}")