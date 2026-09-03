{ inputs, pkgs, ... }:
let
  nativeDeps = [ ];
  pname = baseNameOf ./.;
  python = pkgs.python3;
  pythonDeps = [
    (python.pkgs.nibabel.overridePythonAttrs (_oldAttrs: {
      doCheck = false;
      doInstallCheck = false;
      pytestCheckPhase = "";
    }))
    inputs.self.packages.${pkgs.stdenv.system}.segmentation_models_pytorch
    pkgs.texlive.combined.scheme-full
    python.pkgs.fvcore
    python.pkgs.gdown
    python.pkgs.matplotlib
    python.pkgs.pandas
    python.pkgs.scikit-image
    python.pkgs.torch-bin
    python.pkgs.torchvision-bin
  ];
  shellHook = "";
in
python.pkgs.buildPythonPackage {
  inherit pname;
  inherit shellHook;
  installPhase = ''
    install -Dm644 main.py "$out/${python.sitePackages}/$pname.py"
    install -Dm755 main.py "$out/bin/$pname"
    if [ -d prm ]; then
      cp -R prm/ "$out/${python.sitePackages}/"
      cp -R prm/ "$out/bin/"
    fi
  '';
  meta.mainProgram = pname;
  nativeBuildInputs = nativeDeps;
  passthru.python = python;
  propagatedBuildInputs = pythonDeps;
  pyproject = false;
  src = ./.;
  strictDeps = true;
  version = "0.0.0";
}
