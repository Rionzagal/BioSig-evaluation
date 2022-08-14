from importlib.metadata import entry_points
import os
from subprocess import run, PIPE
from setuptools import setup, find_packages


if __name__ == "__main__":
    pkg_version = (
        run(["git", "describe", "--tags"], stdout=PIPE)
        .stdout.decode("utf-8")
        .strip()
    )

    if "-" in pkg_version:
        v, i, s = pkg_version.split('-')
        pkg_version = f"{v}+{i}.git.{s}"

    assert '-' not in pkg_version
    assert '.' in pkg_version
    assert os.path.isfile("src/medsig/version.py")

    with open("src/medsig/VERSION", "w", encoding="utf-8") as file:
        file.write("%s\n" % pkg_version)

    with open("README.md", "r", encoding="utf-8") as file:
        long_description = file.read()

    setup(
        name="med-signal",
        version=pkg_version,
        author="Rionzagal (Mario González)",
        author_email="mario.gzz.gal@gmail.com",
        license="GPL-3.0",
        license_content_type="text",
        description="A biological signal simulation and evaluation package",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/Rionzagal/MedSig",
        packages=find_packages("src"),
        package_data={"medsig": ["VERSION"]},
        include_package_data=True,
        keywords=["python", "biology", "medical signal", "biological signal"],
        classifiers=[
            "Development Status :: 1 - Planning",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Healthcare Industry",
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
        python_requires=">=3.10",
        install_requires=[
            "numpy>=1.20",
            "pandas>=1.4",
            "scipy>=1",
            "matplotlib>=3",
        ],
        zip_safe=False
    )
