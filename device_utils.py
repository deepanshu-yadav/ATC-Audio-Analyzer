"""
Device and binary discovery utilities for CrispASR pipeline.
Provides cross-platform (Windows & Linux) detection for CPU and CUDA binaries and models.
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List


def get_project_dir() -> Path:
    """
    Returns the absolute path to the project directory.
    
    Returns:
        Path: Absolute path to the directory containing this file.
    """
    return Path(__file__).resolve().parent


# For backward compatibility if imported elsewhere
get_backend_dir = get_project_dir


def get_platform_name() -> str:
    """
    Detect operating system platform.
    
    Returns:
        str: 'windows', 'linux', or 'darwin'.
    """
    if sys.platform.startswith("win"):
        return "windows"
    elif sys.platform.startswith("linux"):
        return "linux"
    elif sys.platform.startswith("darwin"):
        return "darwin"
    return sys.platform


def _get_candidate_binary_names() -> List[str]:
    """
    Get list of candidate executable file names based on OS.
    
    Returns:
        List[str]: Candidate file names.
    """
    if sys.platform.startswith("win"):
        return ["crispasr.exe", "crispasr"]
    else:
        return ["crispasr", "crispasr.bin", "crispasr.exe"]


def _search_in_directory(dir_path: Path, names: List[str]) -> Optional[Path]:
    """
    Search for any candidate binary name in a specific directory.
    
    Args:
        dir_path (Path): Directory to search in.
        names (List[str]): List of candidate file names.
        
    Returns:
        Optional[Path]: Path to executable if found, else None.
    """
    if not dir_path.is_dir():
        return None
    for name in names:
        p = dir_path / name
        if p.is_file():
            # Ensure Linux/Unix executable permission
            if not sys.platform.startswith("win"):
                try:
                    if not os.access(str(p), os.X_OK):
                        os.chmod(str(p), 0o755)
                except Exception as perm_err:
                    print(f"Notice: Could not set executable permission on {p}: {perm_err}")
            return p
    return None


def find_crispasr_binary(device: str = "auto") -> Tuple[str, str, bool]:
    """
    Locates the crispasr binary looking in 'cpu' or 'cuda' folders,
    supporting both Windows and Linux environments.
    
    Args:
        device (str): Requested device ('cuda', 'cpu', or 'auto').
        
    Returns:
        Tuple[str, str, bool]:
            - exe_path: Resolved absolute path to the executable.
            - resolved_device: Actual device used ('cuda' or 'cpu').
            - use_gpu: Boolean flag for GPU usage (True for CUDA, False for CPU).
            
    Raises:
        FileNotFoundError: If no crispasr binary can be found in any candidate folder.
    """
    project_dir = get_project_dir()
    cwd = Path.cwd()
    candidate_names = _get_candidate_binary_names()
    
    device_lower = (device or "auto").strip().lower()
    
    # Candidate directories
    cuda_dirs = [
        project_dir / "bin" / "cuda",
        cwd / "bin" / "cuda",
    ]
    cpu_dirs = [
        project_dir / "bin" / "cpu",
        cwd / "bin" / "cpu",
    ]
    fallback_dirs = [
        project_dir / "bin",
        cwd / "bin",
    ]
    
    cuda_path: Optional[Path] = None
    for d in cuda_dirs:
        cuda_path = _search_in_directory(d, candidate_names)
        if cuda_path:
            break
            
    cpu_path: Optional[Path] = None
    for d in cpu_dirs:
        cpu_path = _search_in_directory(d, candidate_names)
        if cpu_path:
            break
            
    fallback_path: Optional[Path] = None
    for d in fallback_dirs:
        fallback_path = _search_in_directory(d, candidate_names)
        if fallback_path:
            break
            
    # System PATH lookup
    path_binary: Optional[Path] = None
    for name in candidate_names:
        found_in_path = shutil.which(name)
        if found_in_path:
            path_binary = Path(found_in_path).resolve()
            break

    if device_lower == "cuda":
        if cuda_path:
            return str(cuda_path.resolve()), "cuda", True
        elif cpu_path:
            print(f"[Device Selection] Requested 'cuda', but no binary found in 'bin/cuda'. Falling back to CPU binary at: {cpu_path}")
            return str(cpu_path.resolve()), "cpu", False
        elif fallback_path:
            print(f"[Device Selection] Requested 'cuda', using binary found at: {fallback_path}")
            return str(fallback_path.resolve()), "cuda", True
        elif path_binary:
            print(f"[Device Selection] Requested 'cuda', using binary found in system PATH: {path_binary}")
            return str(path_binary), "cuda", True
        else:
            raise FileNotFoundError(
                f"CrispASR binary not found for device 'cuda'. Please place the executable in "
                f"'{project_dir / 'bin' / 'cuda'}' or '{project_dir / 'bin' / 'cpu'}'."
            )

    elif device_lower == "cpu":
        if cpu_path:
            return str(cpu_path.resolve()), "cpu", False
        elif cuda_path:
            print(f"[Device Selection] Requested 'cpu', using binary in 'bin/cuda' with GPU disabled: {cuda_path}")
            return str(cuda_path.resolve()), "cpu", False
        elif fallback_path:
            return str(fallback_path.resolve()), "cpu", False
        elif path_binary:
            return str(path_binary), "cpu", False
        else:
            raise FileNotFoundError(
                f"CrispASR binary not found for device 'cpu'. Please place the executable in "
                f"'{project_dir / 'bin' / 'cpu'}'."
            )

    else:  # 'auto'
        if cuda_path:
            return str(cuda_path.resolve()), "cuda", True
        elif cpu_path:
            return str(cpu_path.resolve()), "cpu", False
        elif fallback_path:
            return str(fallback_path.resolve()), "cpu", False
        elif path_binary:
            return str(path_binary), "cpu", False
        else:
            raise FileNotFoundError(
                f"CrispASR binary not found in 'bin/cpu', 'bin/cuda', or 'bin'. "
                f"Please ensure crispasr executable exists in {project_dir / 'bin' / 'cpu'} or {project_dir / 'bin' / 'cuda'}."
            )


def find_model_path(requested_path: Optional[str] = None) -> str:
    """
    Locates the speech model GGUF file across standard locations.
    
    Args:
        requested_path (Optional[str]): Explicit path provided by caller.
        
    Returns:
        str: Absolute path to the model file.
        
    Raises:
        FileNotFoundError: If no GGUF model file can be found.
    """
    project_dir = get_project_dir()
    cwd = Path.cwd()
    
    # 1. If explicit path exists, return it
    if requested_path:
        p = Path(requested_path)
        if not p.is_absolute():
            # Check relative to cwd and relative to project_dir
            if (cwd / p).is_file():
                return str((cwd / p).resolve())
            if (project_dir / p).is_file():
                return str((project_dir / p).resolve())
        elif p.is_file():
            return str(p.resolve())
            
    # 2. Check standard candidate paths
    candidates = [
        project_dir / "models" / "speech-model.gguf",
        project_dir / "models" / "ggufs" / "speech-model.gguf",
        cwd / "models" / "speech-model.gguf",
        cwd / "models" / "ggufs" / "speech-model.gguf",
    ]
    for c in candidates:
        if c.is_file():
            return str(c.resolve())
            
    # 3. Check for any .gguf in models directory
    models_dir = project_dir / "models"
    if models_dir.is_dir():
        for f in models_dir.glob("*.gguf"):
            if f.is_file():
                return str(f.resolve())
        ggufs_dir = models_dir / "ggufs"
        if ggufs_dir.is_dir():
            for f in ggufs_dir.glob("*.gguf"):
                if f.is_file():
                    return str(f.resolve())
                    
    # Return default path if not found
    fallback = project_dir / "models" / "speech-model.gguf"
    return str(fallback)


def get_subprocess_env(exe_path: str) -> Dict[str, str]:
    """
    Prepares environment variables for running the crispasr subprocess.
    Ensures dynamic libraries (.dll on Windows, .so on Linux) located
    in the binary's folder are discovered by the dynamic loader.
    
    Args:
        exe_path (str): Absolute path to the executable.
        
    Returns:
        Dict[str, str]: Subprocess environment dictionary.
    """
    env = os.environ.copy()
    exe_dir = str(Path(exe_path).resolve().parent)
    
    if sys.platform.startswith("win"):
        # Prepend directory to PATH for Windows DLLs (e.g. openblas.dll)
        current_path = env.get("PATH", "")
        env["PATH"] = exe_dir + os.pathsep + current_path
    else:
        # Prepend directory to LD_LIBRARY_PATH for Linux shared objects
        current_ld = env.get("LD_LIBRARY_PATH", "")
        if current_ld:
            env["LD_LIBRARY_PATH"] = exe_dir + os.pathsep + current_ld
        else:
            env["LD_LIBRARY_PATH"] = exe_dir
            
    return env


def get_system_capabilities() -> Dict[str, Any]:
    """
    Inspects available hardware and binary directories.
    
    Returns:
        Dict[str, Any]: Summary of OS, CPU binary status, and CUDA binary status.
    """
    project_dir = get_project_dir()
    candidate_names = _get_candidate_binary_names()
    
    has_cuda = _search_in_directory(project_dir / "bin" / "cuda", candidate_names) is not None
    has_cpu = _search_in_directory(project_dir / "bin" / "cpu", candidate_names) is not None
    
    model_found = False
    try:
        m = find_model_path()
        model_found = Path(m).is_file()
    except Exception:
        model_found = False
        
    return {
        "platform": get_platform_name(),
        "is_windows": sys.platform.startswith("win"),
        "is_linux": sys.platform.startswith("linux"),
        "cuda_binary_available": has_cuda,
        "cpu_binary_available": has_cpu,
        "model_available": model_found,
        "default_device": "cuda" if has_cuda else "cpu"
    }
