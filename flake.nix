{
  inputs = {
    canonicalization.url = "github:pbizopoulos/canonicalization";
    nixpkgs.url = "github:NixOS/nixpkgs/9ae611a455b90cf061d8f332b977e387bda8e1ca";
    treefmt-nix = {
      inputs.nixpkgs.follows = "nixpkgs";
      url = "github:numtide/treefmt-nix";
    };
  };
  outputs =
    inputs:
    inputs.canonicalization.blueprint {
      inherit inputs;
      nixpkgs.config.allowUnfree = true;
    }
    // {
      inherit (inputs.canonicalization) formatter;
    };
}
