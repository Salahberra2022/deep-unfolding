from setuptools import setup, find_packages

setup(
    name='unfolding_linear',
    version='0.1.0',
    description='Deep unfolding of iterative methods to solve linear equations',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Salah Berra',
    author_email='salahberra39@gmail.com',
    url='https://github.com/Salahberra2022/deep_unfolding',
    license='GPLv3',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy>=1.20.0',
        'torch>=2.2.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
    ],
)
