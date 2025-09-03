import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

REPO_NAME = "Brain-Tumor-MRI-Classification"
AUTHOR_USER_NAME = "mannantanwar"
SRC_REPO = "brainTumorMRIClassification"
AUTHOR_EMAIL = "mannantanwar@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Brain Tumor MRI Classification App built using python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mannantanwar/Brain-Tumor-MRI-Classification",
    project_urls={
        "Bug_Tracker": "https://github.com/mannantanwar/Brain-Tumor-MRI-Classification/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)
