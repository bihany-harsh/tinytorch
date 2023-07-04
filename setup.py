import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tinytorch",
    version="0.1.0",
    author="Harsh Bihany",
    author_email="harshbihany7@gmail.com",
    description="A tiny version of torch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bihany-harsh/tinytorch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy']
)