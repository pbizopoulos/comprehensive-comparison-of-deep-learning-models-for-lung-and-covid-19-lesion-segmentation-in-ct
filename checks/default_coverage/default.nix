{ inputs, pkgs, ... }:
let
  checkName = baseNameOf ./.;
  dependencyInputs = builtins.concatLists [
    (packageDrv.buildInputs or [ ])
    (packageDrv.checkInputs or [ ])
    (packageDrv.nativeBuildInputs or [ ])
    (packageDrv.nativeCheckInputs or [ ])
    (packageDrv.propagatedBuildInputs or [ ])
    (packageDrv.propagatedNativeBuildInputs or [ ])
  ];
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
