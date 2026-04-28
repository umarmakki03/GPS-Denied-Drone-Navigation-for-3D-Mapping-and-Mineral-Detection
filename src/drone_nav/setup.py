from setuptools import setup
setup(
    name='drone_nav',
    version='0.0.1',
    packages=['drone_nav'],
    install_requires=['setuptools'],
    entry_points={
        'console_scripts': [
            'frontier_explorer  = drone_nav.frontier_explorer:main',
            'waypoint_navigator = drone_nav.waypoint_navigator:main',
            'mineral_explorer   = drone_nav.mineral_explorer:main',
            'spectroscopy       = drone_nav.spectroscopy:main',
            'cave_navigator     = drone_nav.cave_navigator:main',
            'waypoint_recorder  = drone_nav.waypoint_recorder:main',
        ],
    },
)
