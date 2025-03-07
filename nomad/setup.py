from setuptools import setup, find_packages

setup(
    name='nomad',
    version='0.0.0',
    packages=find_packages(),
    install_requires=[
        "numba<0.53",
        "lfads_tf2==0.1.0",
        "protobuf==3.19"
    ],
    author="Systems Neural Engineering Lab",
)