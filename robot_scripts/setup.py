from setuptools import find_packages, setup

package_name = 'scripts'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='devtex',
    maintainer_email='simoneliasc.98@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'end_effector_control = scripts.end_effector_control:main',
            'pick_and_place = scripts.pick_and_place:main',
            'server = scripts.server:main',
            'end_effector_publisher = scripts.end_effector_publisher:main',
        ],
    },
)

