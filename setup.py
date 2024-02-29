from setuptools import find_packages, setup
from typing import List

HPY='-e .'

def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements =[req.replace('\n','') for req in requirements]

        if HPY in requirements:
            requirements.remove(HPY)

    return requirements


setup(
    name='Thyroid_disease_detection',
    version='0.0.2',
    author= 'akshat',
    author_email='chourasiaakshat2@gmail.com',
    install_requires = get_requirements('requirements.txt'),
    packages=find_packages()

)