{
  description = "Zola-based blog's flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
        overlays = [
          (_final: prev: {
            zola = prev.zola.override (old: {
              rustPlatform =
                old.rustPlatform
                // {
                  buildRustPackage = args:
                    old.rustPlatform.buildRustPackage (args
                      // {
                        version = "v0.20.7-cestef";
                        src = prev.fetchFromGitHub {
                          owner = "cestef";
                          repo = "zola";
                          rev = "v0.20.7";
                          sha256 = "sha256-gW2etyi//ok+x/CiBJOX/NLp5aoFIDASMFoaAvgoNOE=";
                        };
                        cargoHash = "sha256-eq0V0M6Y83+9TwCNTf61Me72nsAIiyH+wlaPvy4Je+E=";
                        cargoPatches = [./zola.patch];
                      });
                };
            });
          })
        ];
      };
    in {
      devShells.default = pkgs.mkShell {
        packages = with pkgs; [
          nodejs
          pnpm
          biome
          typst
          zola
        ];
      };
    });
}
