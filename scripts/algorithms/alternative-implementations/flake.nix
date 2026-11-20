{
  description = "Flake for capturing the software environment of the different implementations";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/23.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python-packages = with pkgs.python3Packages; [ numpy scipy mpi4py ];
      in {
        packages = {
          jupycpp = pkgs.dockerTools.buildImage {
            name = "guilloteauq/jupycpp";
            tag = "april24";
            copyToRoot = pkgs.buildEnv {
              name = "image-root";
              paths = with pkgs; [
                (python3.withPackages (ps: python-packages))
                julia-bin
                gcc
                eigen
                pkg-config
                openmpi
                llvmPackages.openmp
                bashInteractive
                coreutils-full
              ];
              pathsToLink = [ "/bin" "/include" "/share" ];
            };
          };
        };
        devShells = {
          default = pkgs.mkShell {
            packages = with pkgs; [
              # Workflow related
              snakemake

              # Experiments related
              (python3.withPackages (ps: python-packages))
              julia-bin
              gcc
              eigen
              pkg-config
              openmpi
              llvmPackages.openmp
            ];
          };
        };
      });
}
