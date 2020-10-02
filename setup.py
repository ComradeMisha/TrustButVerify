from setuptools import setup, find_packages

setup(
    name='alpacka',
    description='Alpacka- internal RL framework',
    version='0.0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'gin-config',
        'gym',
        # (need to lazily define alpacka.envs.Sokoban then)
        'gym_sokoban @ git+ssh://git@gitlab.com/awarelab/gym-sokoban.git',
        'numpy',
        'randomdict',
        'matplotlib',
        'ray==0.8.5',
        'tensorflow>=2.2.0',
        'pygame',
        'imageio',
    ],
    extras_require={
        'mrunner': ['mrunner @ git+https://gitlab.com/awarelab/mrunner.git'],
        'dev': ['pylint==2.4.4', 'pylint_quotes', 'pytest', 'ray[debug]'],
        'tracex': ['flask', 'Pillow'],
    }
)
