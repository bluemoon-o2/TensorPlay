import os
import subprocess
import time
import sys
import shutil
import multiprocessing

def main():
    start_time = time.time()
    cwd = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(cwd, "build")
    
    if os.path.exists(build_dir):
        # We don't delete build dir to allow incremental build, 
        # but for this specific task (fixing env), maybe we should?
        # Let's rely on user deleting it if needed, or cmake handling it.
        pass
    else:
        os.makedirs(build_dir)
        
    print("CMake Configuring & Building..")
    
    # Detect Ninja (Disabled for now as it fails consistently)
    # ninja_path = shutil.which("ninja")
    # generator = "Ninja" if ninja_path else None
    generator = "Visual Studio"
    
    cmake_configure_args = ["cmake", "..", "-DCMAKE_TLS_VERIFY=0"]

    # Check for arguments
    use_cuda = True
    if len(sys.argv) > 1 and "--cpu" in sys.argv:
        use_cuda = False
        print("Building for CPU only (requested via --cpu)...")
        cmake_configure_args.append("-DUSE_CUDA=OFF")
    
    if generator == "Ninja":
        ninja_path = shutil.which("ninja")
        cmake_configure_args.extend(["-GNinja", f"-DCMAKE_MAKE_PROGRAM={ninja_path}"])
        # Explicitly set CUDA compiler for Ninja to avoid detection issues
        if use_cuda:
            nvcc_path = shutil.which("nvcc")
            if nvcc_path:
                 # CMake expects forward slashes
                 nvcc_path = nvcc_path.replace("\\", "/")
                 # IMPORTANT: path contains spaces, must be quoted
                 cmake_configure_args.append(f'-DCMAKE_CUDA_COMPILER="{nvcc_path}"')
    else:
        # Use Visual Studio generator
        # Try to help it find CUDA toolkit if installed but integration is missing
        if use_cuda:
            nvcc_path = shutil.which("nvcc")
            if nvcc_path:
                # nvcc is in bin/, toolkit root is one level up
                cuda_root = os.path.dirname(os.path.dirname(nvcc_path))
                print(f"Detected CUDA Root: {cuda_root}")
                # Pass toolkit root via -T cuda=...
                # IMPORTANT: quote the path if it contains spaces
                cmake_configure_args.extend(["-T", f"cuda=\"{cuda_root}\""])
            else:
                 print("Warning: nvcc not found, but CUDA build not explicitly disabled.")

    # Build args
    n_jobs = str(int(multiprocessing.cpu_count() * 1.2))
    cmake_build_args = ["cmake", "--build", ".", "--config", "Release", "--parallel", n_jobs]
    
    # Environment Setup (Only for Ninja/NMake)
    env_setup_cmd = ""
    if generator == "Ninja" and os.name == 'nt' and 'VCINSTALLDIR' not in os.environ:
        vcvars64 = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
        if os.path.exists(vcvars64):
            print("Using vcvars64.bat environment...")
            env_setup_cmd = f'"{vcvars64}" && '
            
    # Construct commands
    # We chain configure and build to ensure environment persists if using vcvars
    configure_cmd = " ".join(cmake_configure_args)
    build_cmd = " ".join(cmake_build_args)
    
    if env_setup_cmd:
        full_cmd = f'{env_setup_cmd} {configure_cmd} && {build_cmd}'
    else:
        # If no env setup needed (e.g. already in dev shell or linux), run normally
        # We can still chain them
        full_cmd = f'{configure_cmd} && {build_cmd}'
        
    try:
        print(f"Executing: {full_cmd}")
        subprocess.check_call(full_cmd, cwd=build_dir, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        sys.exit(1)
    
    print("\nGenerating stubs..")
    # Generate stubs (produces _C/ directory in current cwd because of -O .)
    # Note: nanobind stubgen uses module name for directory if -P is used
    try:
        env = os.environ.copy()
        env["TENSORPLAY_BUILDING_STUBS"] = "1"
        subprocess.check_call([sys.executable, "-m", "nanobind.stubgen", "-m", "tensorplay._C", "-r", "-O", ".", "-P"], cwd=cwd, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Stub generation failed: {e}")
        # Not fatal, but good to know
    
    # Optional: Build wheel if requested
    if len(sys.argv) > 1 and "--wheel" in sys.argv:
        print("\nBuilding wheel...")
        dist_dir = os.path.join(cwd, "dist", "cpu" if not use_cuda else "cuda")
        if not os.path.exists(dist_dir):
            os.makedirs(dist_dir)
        
        # Use pip wheel to build the wheel into the dist directory
        # We use --no-deps to avoid re-downloading dependencies
        try:
            # For CPU build, we need to pass the same USE_CUDA=OFF flag to scikit-build-core
            env = os.environ.copy()
            if not use_cuda:
                env["SKBUILD_CMAKE_ARGS"] = "-DUSE_CUDA=OFF"
            
            subprocess.check_call([sys.executable, "-m", "pip", "wheel", ".", "--wheel-dir", dist_dir, "--no-deps"], cwd=cwd, env=env)
            print(f"Wheel built successfully in {dist_dir}")
        except subprocess.CalledProcessError as e:
            print(f"Wheel build failed: {e}")
            sys.exit(1)

    # Post-processing: Merge compiled extension into the stub directory to form a proper package
    
    src_stub_dir = os.path.join(cwd, "_C")
    dst_stub_dir = os.path.join(cwd, "tensorplay", "_C")
    
    # 1. Merge Stubs (src_stub_dir -> dst_stub_dir)
    if os.path.exists(src_stub_dir):
        print(f"  - Merging stubs into \"{dst_stub_dir}\" ..")
        if not os.path.exists(dst_stub_dir):
            os.makedirs(dst_stub_dir)
            
        for item in os.listdir(src_stub_dir):
            s = os.path.join(src_stub_dir, item)
            d = os.path.join(dst_stub_dir, item)
            # If destination file exists (e.g. __init__.pyi), overwrite it
            # But DO NOT delete __init__.pyd if it exists in dst but not src
            if os.path.isdir(s):
                # For subdirectories, simple replace (assuming stubs don't have complex subdirs mixed with binaries)
                if os.path.exists(d):
                    shutil.rmtree(d)
                shutil.move(s, d)
            else:
                if os.path.exists(d):
                    os.remove(d)
                shutil.move(s, d)
        
        # Cleanup source stub dir
        try:
            os.rmdir(src_stub_dir)
        except OSError:
            pass # Might not be empty if something weird happened

    # 2. Verify Extension
    # With updated CMake, the extension should be named __init__*.pyd and located in tensorplay/_C
    import glob
    
    # Check for __init__*.pyd/so
    ext_files = glob.glob(os.path.join(dst_stub_dir, "__init__*.pyd")) + \
                glob.glob(os.path.join(dst_stub_dir, "__init__*.so"))
    
    if not ext_files:
        # Fallback/Check: Did CMake output _C*.pyd in tensorplay/_C?
        old_exts = glob.glob(os.path.join(dst_stub_dir, "_C*.pyd")) + \
                   glob.glob(os.path.join(dst_stub_dir, "_C*.so"))
        
        if old_exts:
             print("Found _C*.pyd in target dir, renaming to __init__...")
             for f in old_exts:
                base = os.path.basename(f)
                suffix = base[len("_C"):]
                new_name = "__init__" + suffix
                shutil.move(f, os.path.join(dst_stub_dir, new_name))
             ext_files = glob.glob(os.path.join(dst_stub_dir, "__init__*.pyd"))
        else:
             # Check tensorplay/ root (Old CMake behavior)
             root_exts = glob.glob(os.path.join(cwd, "tensorplay", "_C*.pyd")) + \
                         glob.glob(os.path.join(cwd, "tensorplay", "_C*.so"))
             
             if root_exts:
                 print("Found _C*.pyd in package root, moving to tensorplay/_C/__init__...")
                 if not os.path.exists(dst_stub_dir):
                     os.makedirs(dst_stub_dir)
                     
                 for f in root_exts:
                    base = os.path.basename(f)
                    suffix = base[len("_C"):]
                    new_name = "__init__" + suffix
                    shutil.move(f, os.path.join(dst_stub_dir, new_name))
                 ext_files = glob.glob(os.path.join(dst_stub_dir, "__init__*.pyd"))
    
    if not ext_files:
        print("Error: Could not find compiled extension (__init__*.pyd) in tensorplay/_C/")
        sys.exit(1)
        
    print(f"Found extension: {os.path.basename(ext_files[0])}")

    # Clean up old artifacts in root tensorplay/ just in case
    for f in glob.glob(os.path.join(cwd, "tensorplay", "_C*.pyd")):
        try:
            os.remove(f)
        except:
            pass

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nTensorPlay is ready.")
    print(f"Rebuild spend {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
