{ inputs, pkgs, ... }:
let
  python = pkgs.python3;
in
(inputs.afairesi or inputs.self).lib.mkPythonPackage {
  inherit pkgs;
  executable = true;
  meta = {
    description = "Comprehensive Comparison of Deep Learning Models for Lung and COVID-19 Lesion Segmentation in CT";
    homepage = "https://arxiv.org/abs/2009.06412";
  };
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
}
