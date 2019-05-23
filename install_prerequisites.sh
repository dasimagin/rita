apt-get update
apt-get -y upgrade
apt-get install -y cmake python3-pip gettext
apt-get install -y pkg-config zip g++ zlib1g-dev unzip python3 python
apt-get install -y python3-dev python3-numpy libffi-dev libsdl2-dev libosmesa6-dev
apt-get install libav-tools

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
git clone https://github.com/SokolovRE/lab.git && cd lab
bazel build --python_path=$(which python3) -c opt python/pip_package:build_pip_package
./bazel-bin/python/pip_package/build_pip_package /tmp/dmlab_pkg
pip3 install /tmp/dmlab_pkg/DeepMind_Lab-1.0-py3-none-any.whl --force-reinstall
cd ..
rm -rf lab
