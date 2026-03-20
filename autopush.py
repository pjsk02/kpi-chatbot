import subprocess
from datetime import datetime

def auto_push(message=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    commit_msg = message if message else f"auto-save: {timestamp}"
    
    commands = [
        ["git", "add", "."],
        ["git", "commit", "-m", commit_msg],
        ["git", "push"]
    ]
    
    for cmd in commands:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Issue with '{' '.join(cmd)}':\n{result.stderr}")
            return
        print(result.stdout.strip())
    
    print(f"Pushed: {commit_msg}")

if __name__ == "__main__":
    import sys
    msg = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    auto_push(msg)