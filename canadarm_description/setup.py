from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'canadarm_description'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Meshes
        (os.path.join('share', package_name, 'mesh_txt'), glob('mesh_txt/*.txt')),
        # URDF 
        (os.path.join('share', package_name, 'models/canadarm/urdf'), glob('models/canadarm/urdf/*.urdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='chan',
    maintainer_email='chan@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
