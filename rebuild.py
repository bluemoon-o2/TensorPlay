import os
import subprocess
import time

def main():
    start_time = time.time()
    cwd = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(cwd, "build")
    
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)
        print("Configuring CMake...")
        # Use -A x64 for VS generator if needed, but default is usually fine
        subprocess.check_call(["cmake", ".."], cwd=build_dir)
    
    print("Building project...")
    # --parallel for faster build
    import multiprocessing
    # Recommended: CPU count * 1.2 for better I/O handling
    n_jobs = str(int(multiprocessing.cpu_count() * 1.2))
    subprocess.check_call(["cmake", "--build", ".", "--config", "Release", "--parallel", n_jobs], cwd=build_dir)
    
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Build complete. Total time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
