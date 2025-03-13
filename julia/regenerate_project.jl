using Pkg
Pkg.activate(".")

# Add your dependencies - modify these to match your needs
deps = ["HDF5", "SpecialFunctions"]
for dep in deps
    println("Adding $dep...")
    Pkg.add(dep)
end

println("Resolving dependencies...")
Pkg.resolve()

println("Done! Project files regenerated.")
