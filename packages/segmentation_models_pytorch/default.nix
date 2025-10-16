{
  pkgs ? import <nixpkgs> { },
}:
let
  efficientnet-pytorch = pkgs.python312Packages.buildPythonPackage rec {
    format = "pyproject";
    pname = "efficientnet_pytorch";
    propagatedBuildInputs = [
      pkgs.python312Packages.setuptools
      pkgs.python312Packages.torch-bin
    ];
    pythonImportsCheck = [ pname ];
    src = fetchTarball rec {
      sha256 = "181lhizahwvpv14hqsw9nm6hij45k18h0yxbipvqxibjx0v02azm";
      url = "https://api.github.com/repos/lukemelas/EfficientNet-PyTorch/tarball/master";
    };
    version = "0.7.1";
  };
  pretrainedmodels = pkgs.python312Packages.buildPythonPackage rec {
    format = "pyproject";
    pname = "pretrainedmodels";
    propagatedBuildInputs = [
      pkgs.python312Packages.munch
      pkgs.python312Packages.setuptools
      pkgs.python312Packages.six
      pkgs.python312Packages.torchvision-bin
      pkgs.python312Packages.tqdm
    ];
    pythonImportsCheck = [ pname ];
    src = fetchTarball rec {
      sha256 = "1lgaj4fw7vdcq65qkrbx7si25n9df3nis1micl9bnia5a3jkmbrq";
      url = "https://api.github.com/repos/cadene/pretrained-models.pytorch/tarball/master";
    };
    version = "0.7.4";
  };
  timmWithTorch = pkgs.python312Packages.timm.override {
    torch = pkgs.python312Packages.torch-bin;
    torchvision = pkgs.python312Packages.torchvision-bin;
  };
in
pkgs.python312Packages.buildPythonPackage rec {
  format = "wheel";
  pname = builtins.baseNameOf ./.;
  propagatedBuildInputs = [
    efficientnet-pytorch
    pretrainedmodels
    (timmWithTorch.overrideAttrs (_old: {
      doCheck = false;
      doInstallCheck = false;
      pytestCheckPhase = "";
    }))
  ];
  pythonImportsCheck = [ pname ];
  src = pkgs.python312Packages.fetchPypi rec {
    inherit pname version format;
    dist = python;
    python = "py3";
    sha256 = "w04JBHdxqk3Yh4tPiZ6BJXAM0fj32xbljDcgQVQVGgU=";
  };
  version = "0.5.0";
}
