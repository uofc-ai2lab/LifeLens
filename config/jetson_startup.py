import subprocess
from config.audio_settings import STARTUP_SCRIPT_PATH
from config.logger import root_logger as log

def run_jetson_startup_tasks():
    # """Run Jetson-specific startup tasks via external script."""
    # if not STARTUP_SCRIPT_PATH.exists():
    #     log.warning(f"Startup script not found: {STARTUP_SCRIPT_PATH}")
    #     return
    
    # try:
    #     subprocess.run(["bash", str(STARTUP_SCRIPT_PATH)], check=True)
    # except subprocess.CalledProcessError as e:
    #     log.error(f"Startup tasks failed with exit code {e.returncode}")
    #     raise
    # except Exception as e:
    #     log.error(f"Could not run startup tasks: {e}")
    #     raise
    pass
