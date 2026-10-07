{ inputs, pkgs, ... }:
let
  python = pkgs.python3;
in
(inputs.afairesi or inputs.self).lib.mkPythonPackage {
  inherit pkgs;
  executable = true;
  meta.description = "A Python package.";
  nativeBuildInputs = [ pkgs.texliveFull ];
  propagatedBuildInputs = [
    (python.pkgs.nibabel.overridePythonAttrs (_oldAttrs: {
      doCheck = false;
      doInstallCheck = false;
      pytestCheckPhase = "";
    }))
    inputs.self.packages.${pkgs.stdenv.system}.segmentation_models_pytorch
    python.pkgs.fvcore
    python.pkgs.gdown
    python.pkgs.matplotlib
    python.pkgs.pandas
    python.pkgs.scikit-image
  ];
  src = ./.;
  version = "0.0.0";
}
