from setuptools import setup, find_packages
packages = find_packages()

setup(name='skypy',
      packages=packages,
      package_dir={'': 'src'},
      zip_safe=False,
      package_data={'skypy': ['data/*']})
