{
  description = "BitGN Samples Dev environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
          pkgs = nixpkgs.legacyPackages.${system};
          python = pkgs.python314; # stable since 2015
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
              go
              protobuf
              protoc-gen-go
              python # for SDK
              uv
              buf
          ];

          shellHook = ''
            echo "Rinat Abdullin - BitGN Samples Dev Env"

            export UV_PYTHON="${python}/bin/python"
            export UV_PROJECT_ENVIRONMENT="venv"

            # Let bash expand XDG_CACHE_HOME fallback, not Nix:
            export UV_CACHE_DIR="''${XDG_CACHE_HOME:-$HOME/.cache}/uv"


            echo "Go:       $(go version)"
            echo "Protoc:   $(protoc --version)"
            echo "Python:   $(python --version)"
            echo "  path:   $UV_PYTHON"
            echo "  cache:  $UV_CACHE_DIR"
            echo "  venv:   $UV_PROJECT_ENVIRONMENT"
          '';
        };
      }
    );
}
