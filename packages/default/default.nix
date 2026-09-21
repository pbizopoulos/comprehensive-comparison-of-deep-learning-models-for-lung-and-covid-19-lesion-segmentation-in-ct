{ inputs, pkgs, ... }:
let
  nativeDeps = [ pkgs.texliveFull ];
  pname = baseNameOf ./.;
  python = pkgs.python3;
  pythonDeps = [
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
  shellHook = "";
in
python.pkgs.buildPythonPackage {
  inherit pname;
  inherit shellHook;
  installPhase = ''
    install -Dm644 main.py "$out/${python.sitePackages}/$pname/__init__.py"
    mkdir -p "$out/bin"
    printf '%s\n' '#!${python.interpreter}' "from $pname import main" 'main()' > "$out/bin/$pname"
    chmod 755 "$out/bin/$pname"
    if [ -d prm ]; then
      cp -R prm/ "$out/${python.sitePackages}/$pname/"
    fi
  '';
  meta = {
    description = "A Python package.";
    mainProgram = pname;
  };
  nativeBuildInputs = nativeDeps;
  passthru.python = python;
  propagatedBuildInputs = pythonDeps;
  pyproject = false;
  src = ./.;
  strictDeps = true;
  version = "0.0.0";
}
