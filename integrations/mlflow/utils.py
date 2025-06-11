from pathlib import Path 
import tomllib 
from typing import Dict, List


def get_dependencies_from_pyproject() -> Dict: 
    def get_filenames_in_dir(dir_path: Path): 
        return [f.name for f in dir_path.iterdir() if f.is_file()]
    
    # Find the repo root 
    path = Path(__file__).parent
    files = get_filenames_in_dir(path)
    while "pyproject.toml" not in files:  
        path = path.parent 
        files = get_filenames_in_dir(path)

    with open(path / "pyproject.toml", "rb") as file: 
        pyproject = tomllib.load(file)
    
    project = pyproject.get("project", {})
    dependencies = project.get("dependencies", [])
    python_version = "3.12"
    requires_python = project.get("requires-python")
    if requires_python:
        python_version = requires_python.lstrip(">=~^ ").strip()

    return {
        "dependencies": dependencies,
        "python_version": python_version,
        "pyproject_data": pyproject
    }


def generate_pip_requirements(model_only: bool = True) -> List[str]:
    """Generate pip requirements list
    
    Args:
        model_only: If True, filter to only model-runtime dependencies
                   If False, include all project dependencies
    """
    deps_data = get_dependencies_from_pyproject()
    dependencies = deps_data["dependencies"]
    if model_only: 
        dependencies =  [item for item in dependencies if "mlflow" not in item]
    return dependencies 


def generate_conda_env_from_pyproject(model_only: bool = True) -> dict:
    """Generate conda environment from pyproject.toml
    
    Args:
        model_only: If True, include only model-runtime dependencies
    """
    deps_data = get_dependencies_from_pyproject()
    pip_deps = generate_pip_requirements(model_only=model_only)
    
    return {
        "channels": ["conda-forge", "defaults"], 
        "dependencies": [
            f"python={deps_data['python_version']}", 
            "pip", 
            {
                "pip": pip_deps, 
            }
        ]
    }

if __name__ == "__main__": 
    requirements = generate_pip_requirements()
    breakpoint()