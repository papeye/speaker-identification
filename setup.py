from setuptools import setup

with open("readme.md", "r") as f:
    long_description = f.read()


def parse_requirements(filename):
    with open(filename, encoding="utf-16") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="speaker_identifier",
    version="1.0",
    description="A package for identifying speakers using NN.",
    license="MIT",
    long_description=long_description,
    author="Michal Lukawski, Pawel Pajewski, Michal Ryczkowski, Milena Krolikowska",
    author_email="michal.m.lukawski@gmail.com",
    url="https://github.com/papeye/speaker-identification",
    packages=["speaker_identifier"],  # same as name
    install_requires=parse_requirements("requirements.txt"),
)
