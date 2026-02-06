"""
Unified logging utility for LifeLens.

Provides consistent, clean logging across all modules with minimal color usage.
"""

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Logger:
    """Simple logger with standardized formatting."""
    
    def __init__(self, prefix: str):
        """
        Initialize logger with a prefix (e.g., 'root', 'video', 'audio').
        
        Args:
            prefix: Log prefix string (e.g., '[root]', '[video]', '[audio]')
        """
        self.prefix = prefix
    
    def header(self, message: str):
        """Print a section header (colored)."""
        print(f"\n{bcolors.HEADER}{bcolors.BOLD}{'='*60}{bcolors.ENDC}")
        print(f"{bcolors.HEADER}{bcolors.BOLD}{self.prefix} {message}{bcolors.ENDC}")
        print(f"{bcolors.HEADER}{bcolors.BOLD}{'='*60}{bcolors.ENDC}\n")
    
    def info(self, message: str):
        """Print an info message."""
        print(f"{self.prefix} {message}")
    
    def success(self, message: str):
        """Print a success message (minimal green)."""
        print(f"{self.prefix} {bcolors.OKGREEN}✓{bcolors.ENDC} {message}")
    
    def warning(self, message: str):
        """Print a warning message."""
        print(f"{self.prefix} {bcolors.WARNING}⚠{bcolors.ENDC} {message}")
    
    def error(self, message: str):
        """Print an error message."""
        print(f"{self.prefix} {bcolors.FAIL}✗{bcolors.ENDC} {message}")
    
    def debug(self, message: str):
        """Print a debug message (cyan)."""
        print(f"{self.prefix} {bcolors.OKCYAN}[debug]{bcolors.ENDC} {message}")


# Convenience instances for common modules
root_logger = Logger("[root]")
video_logger = Logger("[video][main]")
audio_logger = Logger("[audio][main]")
