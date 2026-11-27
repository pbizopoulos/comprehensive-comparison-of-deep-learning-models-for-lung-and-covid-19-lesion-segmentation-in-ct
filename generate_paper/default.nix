{ inputs, pkgs, ... }:
let
  python = pkgs.python3;
in
inputs.afairesi.lib.mkPythonPackage {
  inherit pkgs;
  executable = true;
  makeWrapperArgs = [
    "--prefix"
    "PATH"
    ":"
    (pkgs.lib.makeBinPath [ pkgs.texliveFull ])
  ];
  meta = {
    description = "Comprehensive Comparison of Deep Learning Models for Lung and COVID-19 Lesion Segmentation in CT";
    homepage = "https://arxiv.org/abs/2009.06412";
  };
  nativeCheckInputs = [
    pkgs.texliveFull
    python.pkgs.pandas-stubs
    python.pkgs.scipy-stubs
  ];
  propagatedBuildInputs = [
    (python.pkgs.nibabel.overridePythonAttrs { doCheck = false; })
    inputs.self.packages.${pkgs.stdenv.hostPlatform.system}.segmentation_models_pytorch
    python.pkgs.fvcore
    python.pkgs.gdown
    python.pkgs.matplotlib
    python.pkgs.pandas
    python.pkgs.scikit-image
  ];
  src = ./.;
}
