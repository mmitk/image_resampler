import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="image-resampler-mmitk", 
    version="0.0.1",
    author="Michael Mitkov",
    author_email="mitkovmichael@gmail.com",
    description="Package for using imbalanced learn resample strategies on image data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mmitk/image_resampler",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)