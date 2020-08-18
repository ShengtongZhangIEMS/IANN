import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="iann-Shengtong_Zhang", # Replace with your own username
    version="0.0.1",
    author="Shengtong Zhang",
    author_email="shengtongzhang2018@u.northwestern.edu",
    description="IANN for function visualization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ShengtongZhangIEMS/IANN",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)