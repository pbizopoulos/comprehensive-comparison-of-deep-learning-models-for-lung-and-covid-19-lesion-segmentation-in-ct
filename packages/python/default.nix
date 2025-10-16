{
  inputs,
  pkgs ? import <nixpkgs> { },
}:
pkgs.python312Packages.buildPythonPackage rec {
  installPhase = ''mkdir -p $out/bin && cp ./main.py $out/bin/${pname}'';
  meta.mainProgram = pname;
  pname = builtins.baseNameOf src;
  propagatedBuildInputs = [
    inputs.self.packages.${pkgs.stdenv.system}.onnxscript
    inputs.self.packages.${pkgs.stdenv.system}.segmentation_models_pytorch
    pkgs.python312Packages.fvcore
    pkgs.python312Packages.gdown
    pkgs.python312Packages.matplotlib
    (pkgs.python312Packages.nibabel.overridePythonAttrs (_oldAttrs: {
      doCheck = false;
      doInstallCheck = false;
      pytestCheckPhase = "";
    }))
    pkgs.python312Packages.onnx
    pkgs.python312Packages.pandas
    pkgs.python312Packages.scikit-image
    pkgs.python312Packages.torch-bin
    pkgs.python312Packages.torchvision-bin
  ];
  pyproject = false;
  src = ./.;
  version = "0.0.0";
}
