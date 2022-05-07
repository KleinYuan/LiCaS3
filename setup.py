from setuptools import find_packages
from setuptools import setup

package_data = {'':  ['*/*.yaml', '*.yaml']}
required_packages = ['opencv-python>=3.4', 'tensorflow-gpu', 'pyyaml', 'python-box']

setup(
    name='training',
    version='0.1',
    install_requires=required_packages,
    packages=find_packages(),
    include_package_data=False,
    package_data=package_data,
    description='Beta',
)
