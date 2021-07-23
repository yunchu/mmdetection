#!/usr/bin/env bash

work_dir=$(realpath "$(dirname $0)")

venv_dir=$1
if [ -z "$venv_dir" ]; then
  venv_dir=venv
fi

cd ${work_dir}

if [[ -e ${venv_dir} ]]; then
  echo
  echo "Virtualenv already exists. Use command to start working:"
  echo "$ . ${venv_dir}/bin/activate"
  exit
fi

# Create virtual environment
# virtualenv ${venv_dir} -p python3 --prompt="(detection)"
python3 -m venv ${venv_dir} --prompt="(detection)"

. ${venv_dir}/bin/activate

# Get CUDA version.
CUDA_HOME_CANDIDATE=/usr/local/cuda
if [ -z "${CUDA_HOME}" ] && [ -d ${CUDA_HOME_CANDIDATE} ]; then
  echo "Exporting CUDA_HOME as ${CUDA_HOME_CANDIDATE}"
  export CUDA_HOME=${CUDA_HOME_CANDIDATE}
fi

if [ -z ${CUDA_VERSION} ] && [ -e "$CUDA_HOME/version.txt" ]; then
  # Get CUDA version from version.txt file.
  CUDA_VERSION=$(cat $CUDA_HOME/version.txt | sed -e "s/^.*CUDA Version *//" -e "s/ .*//")
fi

if [[ -z ${CUDA_VERSION} ]]; then
  # Get CUDA version from nvidia-smi output.
  CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed -e "s/^.*CUDA Version: *//" -e "s/ .*//")
fi

if [[ -z ${CUDA_VERSION} ]]; then
  echo "Using CUDA_VERSION ${CUDA_VERSION}"
else
  echo "CUDA was not found, installing dependencies in CPU-only mode. If you want to use CUDA, set CUDA_HOME and CUDA_VERSION beforehand."
fi

# Remove dots from CUDA version string, if any.
CUDA_VERSION_CODE=$(echo ${CUDA_VERSION} | sed -e "s/\.//" -e "s/\(...\).*/\1/")

# install PyTorch and MMCV.
export TORCH_VERSION=1.8.1
export TORCHVISION_VERSION=0.9.1
export MMCV_VERSION=1.3.0

if [[ -z $CUDA_VERSION_CODE ]]; then
  pip install torch==${TORCH_VERSION}+cpu torchvision==${TORCHVISION_VERSION}+cpu -f https://download.pytorch.org/whl/torch_stable.html
elif [[ $CUDA_VERSION_CODE == "102" ]]; then
  pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION}
else
  pip install torch==${TORCH_VERSION}+cu${CUDA_VERSION_CODE} torchvision==${TORCHVISION_VERSION}+cu${CUDA_VERSION_CODE} -f https://download.pytorch.org/whl/torch_stable.html
fi

if [[ -z $CUDA_VERSION_CODE ]]; then
  pip install --no-cache-dir mmcv-full==${MMCV_VERSION} -f https://download.openmmlab.com/mmcv/dist/cpu/torch${TORCH_VERSION}/index.html
else
  pip install --no-cache-dir mmcv-full==${MMCV_VERSION} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA_VERSION_CODE}/torch${TORCH_VERSION}/index.html
fi

# Install other requirements.
cat requirements.txt | xargs -n 1 -L 1 pip3 install

# path_openvino_vars="${INTEL_OPENVINO_DIR:-/opt/intel/openvino_2021}/bin/setupvars.sh"
# if [[ -e "${path_openvino_vars}" ]]; then
#   echo ". ${path_openvino_vars}" >> ${venv_dir}/bin/activate
# fi

# mo_requirements_file="${INTEL_OPENVINO_DIR:-/opt/intel/openvino_2021}/deployment_tools/model_optimizer/requirements_onnx.txt"
# if [[ -e "${mo_requirements_file}" ]]; then
#   pip install -qr ${mo_requirements_file}
# else
#   echo "[WARNING] Model optimizer requirements were not installed. Please install the OpenVino toolkit to use one."
# fi

pip install -e .
MMDETECTION_DIR=`realpath .`
echo "export MMDETECTION_DIR=${MMDETECTION_DIR}" >> ${venv_dir}/bin/activate

deactivate

echo
echo "Activate a virtual environment to start working:"
echo "$ . ${venv_dir}/bin/activate"
