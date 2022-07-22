import setuptools

with open("README.md", "r") as fh:
	long_description = fh.read()

setuptools.setup(
	name="proSVD",
	version='0.0.1',
	author="Pranjal Gupta, Anne Draelos",
	author_email="pranjal.gupta@duke.edu",
	description="Streaming dimension reduction tools for neural data",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/pearsonlab/proSVD",
	packages=setuptools.find_packages(),

)