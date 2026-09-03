{
  pkgs ? import <nixpkgs> { },
}:
let
  efficientnet-pytorch = python.pkgs.buildPythonPackage rec {
    format = "pyproject";
    pname = "efficientnet_pytorch";
    propagatedBuildInputs = [
      python.pkgs.setuptools
      python.pkgs.torch-bin
    ];
    pythonImportsCheck = [ pname ];
    src = fetchTarball rec {
      sha256 = "181lhizahwvpv14hqsw9nm6hij45k18h0yxbipvqxibjx0v02azm";
      url = "https://api.github.com/repos/lukemelas/EfficientNet-PyTorch/tarball/master";
    };
    version = "0.7.1";
  };
  pretrainedmodels = python.pkgs.buildPythonPackage rec {
    format = "pyproject";
    pname = "pretrainedmodels";
    propagatedBuildInputs = [
      python.pkgs.munch
      python.pkgs.setuptools
      python.pkgs.six
      python.pkgs.torchvision-bin
      python.pkgs.tqdm
    ];
    pythonImportsCheck = [ pname ];
    src = fetchTarball rec {
      sha256 = "1lgaj4fw7vdcq65qkrbx7si25n9df3nis1micl9bnia5a3jkmbrq";
      url = "https://api.github.com/repos/cadene/pretrained-models.pytorch/tarball/master";
    };
    version = "0.7.4";
  };
  python = pkgs.python3;
  timmWithTorch = python.pkgs.timm.override {
    torch = python.pkgs.torch-bin;
    torchvision = python.pkgs.torchvision-bin;
  };
in
python.pkgs.buildPythonPackage rec {
  format = "wheel";
  pname = builtins.baseNameOf ./.;
  propagatedBuildInputs = [
    (timmWithTorch.overrideAttrs (_old: {
      doCheck = false;
      doInstallCheck = false;
      pytestCheckPhase = "";
    }))
    efficientnet-pytorch
    pretrainedmodels
  ];
  pythonImportsCheck = [ pname ];
  src = python.pkgs.fetchPypi rec {
    inherit pname version format;
    dist = python;
    python = "py3";
    sha256 = "w04JBHdxqk3Yh4tPiZ6BJXAM0fj32xbljDcgQVQVGgU=";
  };
  version = "0.5.0";
}
