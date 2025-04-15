from setuptools import setup, find_packages

setup(
    name='mlc_adaptive_threshold',
    version='0.0.1',
    packages=find_packages(),
    #include_package_data=True,  # This ensures MANIFEST.in is used if it exists
    #package_data={
    #    'omo.toy_icd10cm_encoder': ['data/*'],  # Corrected path to include data
    #},
)
