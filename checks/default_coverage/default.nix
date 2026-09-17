{ inputs, pkgs, ... }:
let
  checkName = baseNameOf ./.;
  dependencyInputs = pkgs.lib.concatMap (name: packageDrv.${name} or [ ]) [
    "buildInputs"
    "checkInputs"
    "nativeBuildInputs"
    "nativeCheckInputs"
    "propagatedBuildInputs"
    "propagatedNativeBuildInputs"
  ];
  packageDrv = inputs.self.packages.${pkgs.stdenv.system}.${packageName};
  packageName = pkgs.lib.removeSuffix "_coverage" checkName;
  pythonEnv = packageDrv.python.withPackages (
    ps:
    packageDrv.propagatedBuildInputs
    ++ [
      ps.hypothesis
      ps.pytest
      ps.pytest-cov
    ]
  );
in
pkgs.runCommand checkName
  {
    inherit (packageDrv) src;
    nativeBuildInputs = dependencyInputs ++ [ pythonEnv ];
  }
  ''
    export HOME="$(mktemp -d)"
    mkdir -p "$out/html" packages
    ln -s "$src" "packages/${packageName}"
    export PYTHONPATH="$PWD:$PYTHONPATH"
    cd "$out"
    PACKAGE_E2E_EXECUTABLE="${packageDrv}/bin/${packageName}" python -c 'import sys; from hypothesis import Phase, settings; settings.register_profile("coverage", phases=[Phase.explicit]); settings.load_profile("coverage"); import pytest; sys.exit(pytest.main(sys.argv[1:]))' -p no:cacheprovider --import-mode=importlib --cov="packages.${packageName}.main" --cov-report "html:$out/html" "$src/test_main.py"
  ''
