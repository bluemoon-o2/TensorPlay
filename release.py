import os
import sys
import importlib
import subprocess
import shutil
import glob
import time
import argparse
import re
from contextlib import contextmanager

import ctypes

def get_short_path_name(long_name):
    """
    Gets the short path name (8.3 format) on Windows to avoid spaces.
    """
    if sys.platform != 'win32':
        return long_name
    if not os.path.exists(long_name):
        return long_name
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # Get required buffer size
        length = kernel32.GetShortPathNameW(long_name, None, 0)
        if length == 0:
            return long_name
        # Get short path
        buffer = ctypes.create_unicode_buffer(length)
        kernel32.GetShortPathNameW(long_name, buffer, length)
        return buffer.value
    except Exception as e:
        print(f"Warning: Failed to get short path for {long_name}: {e}")
        return long_name

def print_step(msg):
    print(f"\n{'='*60}")
    print(f" {msg}")
    print(f"{'='*60}\n")

def run_command(cmd, cwd=None, env=None, shell=False, check=True):
    cmd_str = ' '.join(cmd) if isinstance(cmd, list) else cmd
    print(f"\n>> {cmd_str}\n")
    subprocess.run(cmd, cwd=cwd, env=env, shell=shell, check=check)

def ensure_package(package):
    try:
        importlib.import_module(package)
        print(f"✅ {package}")
    except (ModuleNotFoundError or ImportError) as e:
        print(f"❌ {package} import failed: {e}")
        print(f"  - Try to install {package}...")
        try:
            run_command([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"❌ {package} install failed: {e}")
            sys.exit(1)

@contextmanager
def patch_pyproject_version(version_suffix):
    """
    Temporarily patches pyproject.toml with a version suffix (e.g. +cpu).
    """
    if not version_suffix:
        yield
        return

    pyproject_path = "pyproject.toml"
    with open(pyproject_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Simple regex replace for version
    # Assumes version = "..." is present
    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if not match:
        print("Warning: Could not find version in pyproject.toml")
        yield
        return

    original_version = match.group(1)
    # Check if suffix already exists to avoid +cpu+cpu
    if version_suffix in original_version:
        new_version = original_version
    else:
        new_version = original_version + version_suffix
    
    print(f"  -> Patching version: {original_version} -> {new_version}")
    new_content = content.replace(f'version = "{original_version}"', f'version = "{new_version}"')
    
    with open(pyproject_path, "w", encoding="utf-8") as f:
        f.write(new_content)
        
    try:
        yield
    finally:
        print(f"  -> Restoring version: {original_version}")
        with open(pyproject_path, "w", encoding="utf-8") as f:
            f.write(content)

def get_cuda_version_suffix():
    """
    Detects CUDA version from nvcc and returns a suffix like '+cu121'.
    Returns empty string if CUDA is not found.
    """
    nvcc = shutil.which("nvcc")
    if not nvcc:
        print("Warning: nvcc not found. Cannot determine CUDA version.")
        return ""
    
    try:
        # Run nvcc --version
        output = subprocess.check_output([nvcc, "--version"], encoding="utf-8")
        # Look for "release X.Y"
        match = re.search(r"release (\d+\.\d+)", output)
        if match:
            ver = match.group(1) # e.g. "12.1"
            suffix = "+cu" + ver.replace(".", "")
            print(f"Detected CUDA version: {ver} -> suffix: {suffix}")
            return suffix
        else:
            print("Warning: Could not parse CUDA version from nvcc output.")
            return ""
    except Exception as e:
        print(f"Warning: Failed to run nvcc: {e}")
        return ""

def load_vcvars_env(vcvars_path):
    print(f"Loading environment from {vcvars_path}...")
    # Use "&& set" to print environment after loading
    cmd = f'"{vcvars_path}" > nul && set' 
    try:
        # shell=True is required to run batch file
        output = subprocess.check_output(cmd, shell=True, encoding='utf-8', errors='replace')
        env = {}
        for line in output.splitlines():
            if '=' in line:
                k, v = line.split('=', 1)
                env[k] = v
        return env
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to load vcvars: {e}")
        return {}

def find_vcvars():
    if sys.platform != 'win32':
        return None
        
    roots = [
        r"C:\Program Files\Microsoft Visual Studio",
        r"C:\Program Files (x86)\Microsoft Visual Studio"
    ]
    
    for root in roots:
        if not os.path.exists(root):
            continue
        try:
            for year in os.listdir(root):
                year_path = os.path.join(root, year)
                if not os.path.isdir(year_path):
                    continue
                for edition in os.listdir(year_path):
                    edition_path = os.path.join(year_path, edition)
                    vcvars = os.path.join(edition_path, "VC", "Auxiliary", "Build", "vcvars64.bat")
                    if os.path.exists(vcvars):
                        return vcvars
        except OSError:
            continue
    return None

def main():
    parser = argparse.ArgumentParser(description="Build and release TensorPlay wheels.")
    parser.add_argument("--version", type=str, default="313", help="Python version suffix (e.g. 313, 312)")
    parser.add_argument("--variant", choices=["cpu", "cuda", "all"], default="all", help="Build variant (cpu, cuda, or all)")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running unit tests")
    parser.add_argument("--upload", action="store_true", help="Upload to PyPI after build")
    args = parser.parse_args()

    start_time = time.time()
    
    # 1. Ensure build/test deps
    print_step("Step 0: Checking Dependencies")
    ensure_package("pytest")
    ensure_package("cibuildwheel")
    ensure_package("twine")
        
    # 2. Run Unit Tests (On original source, optional)
    if not args.skip_tests:
        print_step("Step 2: Running Unit Tests")
        
        # Set PYTHONPATH to current directory so 'tensorplay' can be imported
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()
        
        test_dir = "test"
        if not os.path.exists(test_dir):
            print("Warning: 'test' directory not found, skipping tests.")
        else:
            # Run pytest
            print(f"Running tests in {test_dir}...")
            try:
                run_command([sys.executable, "-m", "pytest", "-v", "-s", "-x", test_dir], env=env)
            except subprocess.CalledProcessError:
                print("Tests failed!")
                sys.exit(1)
            print("Tests passed successfully.")

    # 3. Setup Release Directory
    print_step("Step 3: Setting up Release Directory")
    
    cwd = os.getcwd()
    release_dir = os.path.abspath("release_build")
    
    if os.path.exists(release_dir):
        print(f"Cleaning existing release directory: {release_dir}")
        def remove_readonly(func, path, _):
            "Clear the readonly bit and reattempt the removal"
            os.chmod(path, 0o777)
            func(path)
            
        shutil.rmtree(release_dir, onexc=remove_readonly)
    
    print(f"Copying source to {release_dir}...")
    
    # Ignore patterns to avoid copying heavy build artifacts
    def ignore_patterns(path, names):
        ignore_list = []
        if path == cwd:
            ignore_list.extend([
                'dist', 'release_build', 'cmake', 'build', '.gitignore',
                '.git', '.vs', '.vscode', '.trae', '.pytest_cache',
                '__pycache__', 'alpha', 'test', 'benchmark', 'website',
            ])
        return ignore_list

    shutil.copytree(cwd, release_dir, ignore=ignore_patterns)
    print("Source copied.")

    # 4. Build Wheels
    print_step(f"Step 4: Building Wheels ({args.variant})")

    # Create dist dir in original cwd to collect results
    dist_dir = os.path.join(cwd, "dist")
    os.makedirs(dist_dir, exist_ok=True)

    variants = []
    if args.variant == "all":
        variants = ["cpu", "cuda"]
    else:
        variants = [args.variant]

    # Detect CUDA version once if needed
    cuda_suffix = "+cu130"
    nvcc_path = None
    if "cuda" in variants:
        nvcc_path = shutil.which("nvcc")
        cuda_suffix = get_cuda_version_suffix()
        if not cuda_suffix:
            print("Warning: No CUDA version detected, using default suffix or none.")
            pass
    configs = {
        "cpu": {
            "cmake_args": "-DUSE_CUDA=OFF",
            "version_suffix": ""
        },
        "cuda": {
            "cmake_args": "-DUSE_CUDA=ON",
            "version_suffix": cuda_suffix # Main package with version suffix
        }
    }
    
    # Force CMAKE_CUDA_COMPILER if nvcc is found
    if nvcc_path and "cuda" in variants:
        # Use short path to avoid space issues with CMake/Ninja
        nvcc_path = get_short_path_name(nvcc_path)
        nvcc_path = nvcc_path.replace("\\", "/")
        print(f"Forcing CMAKE_CUDA_COMPILER={nvcc_path}")
        configs["cuda"]["cmake_args"] += f";-DCMAKE_CUDA_COMPILER={nvcc_path}"

    try:
        # Switch to release dir
        os.chdir(release_dir)
        
        # Pre-build Step: Copy existing stubs from main directory instead of rebuilding
        # We assume the user has run rebuild.py locally and stubs are present in tensorplay/_C
        print("\n>>> Copying existing stubs from main directory...")
        
        main_stub_dir = os.path.join(cwd, "tensorplay", "_C")
        release_stub_dir = os.path.join(release_dir, "tensorplay", "_C")
        
        if os.path.exists(main_stub_dir):
            # If release_stub_dir exists (from copytree), we can overwrite or merge
            # copytree might have copied it if it wasn't ignored.
            # But let's ensure it's up-to-date with what's in main dir (which contains the latest stubs)
            if os.path.exists(release_stub_dir):
                shutil.rmtree(release_stub_dir)
            
            shutil.copytree(main_stub_dir, release_stub_dir)
            print(f"  - Copied stubs to {release_stub_dir}")
            # Remove any .pyd/.so files that might have been copied
            for f in glob.glob(os.path.join(release_stub_dir, "*.pyd")) + \
                     glob.glob(os.path.join(release_stub_dir, "*.so")) + \
                     glob.glob(os.path.join(release_stub_dir, "*.exp")) + \
                     glob.glob(os.path.join(release_stub_dir, "*.lib")):
                os.remove(f)
                print(f"  - Removed pre-existing binary from stubs: {os.path.basename(f)}")
        else:
            raise FileNotFoundError(f"Warning: Stubs not found in main directory (tensorplay/_C). Wheel might miss type hints.")

        for v in variants:
            cfg = configs[v]
            print(f"\n>>> Building variant: {v} (suffix='{cfg['version_suffix']}')")
            
            cibw_env = os.environ.copy()
            
            # On Windows, ensure we have MSVC environment for Ninja + CUDA
            if sys.platform == 'win32' and (not shutil.which("cl") or not shutil.which("rc")):
                vcvars = find_vcvars()
                if vcvars:
                    print(f"  -> MSVC compiler not found in PATH. Loading {vcvars}...")
                    vs_env = load_vcvars_env(vcvars)
                    if vs_env:
                        cibw_env.update(vs_env)
                        # Verify cl.exe
                        cl_path = shutil.which("cl", path=cibw_env.get("PATH"))
                        if cl_path:
                            print(f"  -> Found MSVC compiler: {cl_path}")
                        else:
                            print("  -> Warning: Loaded vcvars but cl.exe still not found in PATH!")
                    else:
                        print("  -> Warning: Failed to load MSVC environment.")
                else:
                    print("  -> Warning: MSVC compiler (cl.exe) not found and vcvars64.bat not found.")
                    print("     CUDA compilation with Ninja might fail.")

            # Clean tensorplay/lib in release_build to avoid pollution from previous builds
            # or source copy if it wasn't ignored correctly
            lib_dir = os.path.join(release_dir, "tensorplay", "lib")
            if os.path.exists(lib_dir):
                print(f"  - Cleaning {lib_dir} before build...")
                shutil.rmtree(lib_dir)
            
            # Pass CMake args to scikit-build-core
            print(f"  -> Setting SKBUILD_CMAKE_ARGS={cfg['cmake_args']}")
            cibw_env["SKBUILD_CMAKE_ARGS"] = cfg["cmake_args"]
            cibw_env["CMAKE_GENERATOR"] = "Ninja"
            
            # Ensure CUDA_PATH is set for FindCUDAToolkit
            if v == "cuda" and nvcc_path:
                cuda_bin = os.path.dirname(nvcc_path)
                cuda_root = os.path.dirname(cuda_bin)
                if "CUDA_PATH" not in cibw_env:
                    cibw_env["CUDA_PATH"] = cuda_root
                    print(f"  -> Set CUDA_PATH={cuda_root} for build")
                # Also add to PATH to be safe
                if cuda_bin not in cibw_env.get("PATH", ""):
                    cibw_env["PATH"] = cuda_bin + os.pathsep + cibw_env.get("PATH", "")
            
            # Build for cp39, cp310, cp311, cp312, cp313
            # Fix: args.version might be a string like "313" or "310,313"
            # Split correctly
            versions = args.version.split(',')
            cibw_env["CIBW_BUILD"] = ' '.join([f"cp{v}-*" for v in versions])
            cibw_env["CIBW_ARCHS"] = "auto64"
            cibw_env["CIBW_SKIP"] = "*-win32 *musllinux*"
            
            # On Windows, we want to ensure we pick up the right CUDA env if needed
            # but cibuildwheel runs in isolation/docker (on linux) or host (on windows).
            # On Windows host, it inherits env.
            
            with patch_pyproject_version(cfg["version_suffix"]):
                try:
                    run_command([sys.executable, "-m", "cibuildwheel", "--output-dir", dist_dir], env=cibw_env)
                except subprocess.CalledProcessError as e:
                    print(f"❌ cibuildwheel failed for {v}: {e}")
                    sys.exit(1)
    except Exception as e:
        print(f"\n❌ Build process failed: {e}")
        sys.exit(1)
    finally:
        # Restore cwd
        os.chdir(cwd)
        # Cleanup release dir
        if os.path.exists(release_dir):
            print(f"\nCleaning up release directory: {release_dir}")
            try:
                shutil.rmtree(release_dir)
            except Exception as e:
                print(f"Warning: Failed to cleanup release dir: {e}")

    wheels = glob.glob(os.path.join(dist_dir, "*.whl"))
    if not wheels:
        print("Error: No wheels were generated!")
        sys.exit(1)
        
    print(f"\nSuccess! Generated {len(wheels)} wheels:")
    for w in wheels:
        print(f" - {os.path.basename(w)}")

    # 5. Upload to PyPI
    if args.upload:
        print_step("Step 5: Upload to PyPI")
        
        # Check for local versions which PyPI rejects
        local_wheels = [w for w in wheels if "+" in w]
        if local_wheels:
            print("\n⚠️  WARNING: Local version identifiers found (e.g. +cpu, +cu130).")
            print("   PyPI main index DOES NOT support local versions and will reject these uploads.")
            print("   See: https://packaging.python.org/specifications/core-metadata")
            print("   Files affected:")
            for w in local_wheels:
                print(f"    - {os.path.basename(w)}")
            print("\n   Continuing upload (expect failures)...")

        print("Uploading to PyPI...")
        try:
            run_command([sys.executable, "-m", "twine", "upload", os.path.join(dist_dir, "*")])
        except subprocess.CalledProcessError:
            print("\n❌ Upload failed. This is expected for local versions (+cpu, +cuXX) on PyPI.")
            print("   Solution: Use a private index, GitHub Releases, or strip the suffix for the main release.")
    else:
        print("\nSkipping upload. Use --upload to upload artifacts.")

    elapsed = time.time() - start_time
    print_step(f"Done! Total time: {elapsed:.2f}s")

if __name__ == "__main__":
    main()
