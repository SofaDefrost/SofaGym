from setuptools import setup, find_packages

setup(
    name="sofagym",
    version="0.0.1",
    description="An environment based on Sofa",
    install_requires=["gymnasium>=1.0", "numpy", "glfw", "pygame"],
    authors=["Etienne Ménager", "Pierre Schegg"],
    authors_email=["pierre.schegg@robocath.com", "etienne.menager@ens-rennes.fr"],
    keywords="simulation environment reinforcement learning SOFA",
    packages=find_packages(include=["sofagym"], exclude=["tests"]),
    entry_points={
        "console_scripts": [],
    },
)
