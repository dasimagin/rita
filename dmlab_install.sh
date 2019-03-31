export PYTHON_BIN_PATH="/usr/bin/python3"
git clone https://github.com/deepmind/lab.git && cd lab
echo "
# Description:
#   Build rule for Python and Numpy.
#   This rule works for Debian and Ubuntu. Other platforms might keep the
#   headers in different places, cf. 'How to build DeepMind Lab' in build.md.

cc_library(
    name = \"python\",
    hdrs = glob([
        \"include/python3.6/*.h\",
        \"lib/python3/dist-packages/numpy/core/include/numpy/*.h\",
    ]),
    includes = [
        \"include/python3.6\",
        \"lib/python3/dist-packages/numpy/core/include\",
    ],
    visibility = [\"//visibility:public\"],
)
" > python.BUILD
cd python/pip_package
echo "
import setuptools

REQUIRED_PACKAGES = [
    'numpy >= 1.13.3',
    'six >= 1.10.0',
]

setuptools.setup(
    name='DeepMind Lab',
    version='1.0',
    description='DeepMind Lab',
    long_description='',
    url='https://github.com/deepmind/lab',
    author='DeepMind',
    packages=setuptools.find_packages(),
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True)
" > setup.py
cd ../..

bazel build --python_path=/usr/bin/python3 -c opt python/pip_package:build_pip_package
./bazel-bin/python/pip_package/build_pip_package /tmp/dmlab_pkg
pip install /tmp/dmlab_pkg/DeepMind_Lab-1.0-py3-none-any.whl --force-reinstall
