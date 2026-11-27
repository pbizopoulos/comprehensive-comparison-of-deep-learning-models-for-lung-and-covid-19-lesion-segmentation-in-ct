{ pkgs }:
pkgs.python3.pkgs.buildPythonPackage rec {
  dependencies = with pkgs.python3.pkgs; [
    huggingface-hub
    numpy
    pillow
    safetensors
    timm
    torch
    torchvision
    tqdm
  ];
  format = "wheel";
  meta.description = "Image segmentation models with pretrained encoders for PyTorch";
  pname = builtins.baseNameOf ./.;
  pythonImportsCheck = [ pname ];
  src = pkgs.fetchPypi {
    inherit pname version format;
    dist = "py3";
    hash = "sha256-xGFiIhtavjvClrIi++qKqA0LVucTD33e9lT3B4BFNOI=";
    python = "py3";
  };
  version = "0.6.0";
}
