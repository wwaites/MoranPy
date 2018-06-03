from setuptools import setup

setup(
    name='pdkq',
    version='1.0',
    packages=['pd'],
    install_requires=['numpy'],
    entry_points={
        'console_scripts': [
            'pdsim = pd.simulation:main',
            'uvroc = pd.uv:main',
            'mvroc = pd.mv:main'
            ]
        }
    )
