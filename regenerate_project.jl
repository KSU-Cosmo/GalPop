using Pkg
Pkg.activate(".")

# Add your dependencies here - customize this list
# based on what your project actually needs
pkgs = ["HDF5"]
for pkg in pkgs
    @info "Adding $pkg"
    Pkg.add(pkg)
end

# Make sure Test is available for testing
# This way works across Julia versions
Pkg.add("Test")

# Resolve all dependencies
Pkg.resolve()
@info "Project files regenerated successfully"
