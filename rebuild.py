import os
import subprocess


def main():
    cwd = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(cwd, "build")
    
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)
        print("Configuring CMake...")
        # Use -A x64 for VS generator if needed, but default is usually fine
        subprocess.check_call(["cmake", ".."], cwd=build_dir)
    
    print("Building project...")
    # --parallel for faster build
    subprocess.check_call(["cmake", "--build", ".", "--config", "Release", "--parallel", "4"], cwd=build_dir)
    
    print("Build complete.")

if __name__ == "__main__":
    main()
