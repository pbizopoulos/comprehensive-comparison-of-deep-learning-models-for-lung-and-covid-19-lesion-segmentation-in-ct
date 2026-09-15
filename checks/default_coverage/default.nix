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
    mkdir -p "$out/html"
    cd "$out"
    PACKAGE_E2E_EXECUTABLE="${packageDrv}/bin/${packageName}" python -m pytest -p no:cacheprovider --cov="$src" --cov-report "html:$out/html" "$src/main.py"
  ''
