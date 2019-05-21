export PYTHON_BIN_PATH=$(which python3)
echo "PYTHON_BIN_PATH=$(which python3)" >> ~/.bashrc
git clone https://github.com/SokolovRE/lab.git && cd lab
bazel build --python_path=$(which python3) -c opt python/pip_package:build_pip_package
./bazel-bin/python/pip_package/build_pip_package /tmp/dmlab_pkg
pip3 install /tmp/dmlab_pkg/DeepMind_Lab-1.0-py3-none-any.whl --force-reinstall
cd ..
rm -rf lab
