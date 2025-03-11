using Documenter
using GalPop

DocMeta.setdocmeta!(GalPop, :DocTestSetup, :(using GalPop); recursive=true)

makedocs(;
    modules=[GalPop],
    authors="KSU-Cosmo",
    repo="https://github.com/KSU-Cosmo/GalPop/blob/{commit}{path}#{line}",
    sitename="GalPop.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://KSU-Cosmo.github.io/GalPop.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API Reference" => "api.md",
        "Examples" => "examples.md",
    ],
)

deploydocs(;
    repo="github.com/KSU-Cosmo/GalPop.jl",
    devbranch="main",
)