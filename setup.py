from setuptools import find_packages, setup
from typing import List

HIPHEN_E_DOT = "-e ."

def get_requirements(filepath : str) -> List[str]:
    requirements = []
    
    with open (filepath) as file_obj:
        requirements = file_obj.readlines()
        requirements = [i.replace("\n", "") for i in requirements]
        
        if HIPHEN_E_DOT in requirements:
            requirements.remove(HIPHEN_E_DOT)

setup(
    name ="ML_Pipeleine_Project",
    version='0.0.1',
    author='Supratim Mukherjee',
    description="Machine Learning Pipeline Project",
    author_email="supratim.2127@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements("requirements.txt")
)