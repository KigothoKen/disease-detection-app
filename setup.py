from setuptools import setup, find_packages

setup(
    name="disease-detection-app",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "flask==2.3.3",
        "pillow==10.1.0",
        "numpy==1.24.3",
        "tensorflow==2.12.0",
        "tflite-runtime==2.12.0",
        "werkzeug==2.3.7"
    ],
    include_package_data=True,
    zip_safe=False
)
