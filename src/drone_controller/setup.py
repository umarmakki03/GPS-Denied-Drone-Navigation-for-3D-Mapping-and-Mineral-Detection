from setuptools import setup
setup(
    name='drone_controller',
    version='0.0.1',
    packages=['drone_controller'],
    install_requires=['setuptools'],
    entry_points={
        'console_scripts': [
            'hover_controller = drone_controller.hover_controller:main',
            'drone_teleop = drone_controller.drone_teleop:main',
        ],
    },
)
