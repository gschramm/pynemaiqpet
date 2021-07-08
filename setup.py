import setuptools

setuptools.setup(
    name="pynemaiqpet",
    use_scm_version={'fallback_version':'unkown'},
    setup_requires=['setuptools_scm','setuptools_scm_git_archive'],
    author="Georg Schramm",
    author_email="georg.schramm@kuleuven.be",
    description="Analysis of PET NEMA IQ phantom scans",
    long_description="Analysis of PET NEMA IQ phantom scans",
    license='MIT',
    long_description_content_type="text/markdown",
    url="https://github.com/gschramm/pynemaiqpet",
    packages=setuptools.find_packages(exclude=["data","wip"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6, <3.9',
    install_requires=['pymirc>=0.23',
                      'nibabel>=3.0',
                      'matplotlib>=3.1',
                      'pydicom>=2.0',
                      'scipy>=1.4',
                      'pandas>=1.0',
                      'lmfit>=1.0'],
    entry_points = {'console_scripts' : ['pynemaiqpet_wb_nema_iq=pynemaiqpet.command_line_tools:wb_nema_iq'],},
)
