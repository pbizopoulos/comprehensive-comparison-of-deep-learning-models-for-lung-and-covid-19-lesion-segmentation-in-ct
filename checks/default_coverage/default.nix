{ inputs, pkgs, ... }:
let
  checkName = baseNameOf ./.;
  dependencyInputs = builtins.concatLists (
    builtins.attrValues (
      pkgs.lib.filterAttrs (
        name: _:
        builtins.elem name [
          "buildInputs"
          "checkInputs"
          "nativeBuildInputs"
          "nativeCheckInputs"
          "propagatedBuildInputs"
          "propagatedNativeBuildInputs"
        ]
      ) packageDrv
    )
  );
  packageDrv = inputs.self.packages.${pkgs.stdenv.system}.${packageName};
  packageName = pkgs.lib.removeSuffix "_coverage" checkName;
  pythonEnv = packageDrv.python.withPackages (
    _:
    packageDrv.propagatedBuildInputs
    ++ [
      packageDrv.python.pkgs.pytest
      packageDrv.python.pkgs.pytest-cov
    ]
  );
in
pkgs.runCommand checkName
  {
    nativeBuildInputs = dependencyInputs ++ [ pythonEnv ];
    src = ../.. + "/packages/${packageName}";
  }
  ''
    export HOME="$(mktemp -d)"
    mkdir -p "$out/html" packages
    ln -s "$src" "packages/${packageName}"
    export PYTHONPATH="$PWD:$PYTHONPATH"
    cd "$out"
    PACKAGE_E2E_EXECUTABLE="${packageDrv}/bin/${packageName}" python -m pytest -p no:cacheprovider --import-mode=importlib --cov="packages.${packageName}.main" --cov-report "html:$out/html" "$src/test_main.py"
  ''
