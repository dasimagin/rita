sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get install -y cmake python3-pip gettext
sudo apt-get install -y pkg-config zip g++ zlib1g-dev unzip python3 python
sudo apt-get install -y python3-dev python3-numpy libffi-dev libsdl2-dev libosmesa6-dev

pip3 install torch
pip3 install torchvision

wget https://github.com/bazelbuild/bazel/releases/download/0.22.0/bazel-0.22.0-installer-linux-x86_64.sh
chmod +x bazel-0.22.0-installer-linux-x86_64.sh
./bazel-0.22.0-installer-linux-x86_64.sh --user
echo "export PATH=\"$PATH:$HOME/bin\"" >> ~/.bashrc
export PATH="$PATH:$HOME/bin"
rm bazel-0.22.0-installer-linux-x86_64.sh

pip3 install gym
cd ..
git clone https://github.com/openai/gym.git
cd gym
pip3 install -e '.[atari]'
cd ../rita

export PYTHON_BIN_PATH=$(which python3)
echo "PYTHON_BIN_PATH=$(which python3)" >> ~/.bashrc
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

bazel build --verbose_failures --python_path=$(which python3) -c opt python/pip_package:build_pip_package
./bazel-bin/python/pip_package/build_pip_package /tmp/dmlab_pkg
pip3 install /tmp/dmlab_pkg/DeepMind_Lab-1.0-py3-none-any.whl --force-reinstall
cd ..
rm -rf lab
