using Documenter
using GalPop

DocMeta.setdocmeta!(GalPop, :DocTestSetup, :(using GalPop); recursive=true)

makedocs(;
    modules=[GalPop],
    authors="Your Name",
    repo="https://github.com/yourusername/GalPop.jl/blob/{commit}{path}#{line}",
    sitename="GalPop.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://yourusername.github.io/GalPop.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API Reference" => "api.md",
        "Examples" => "examples.md",
    ],
)

deploydocs(;
    repo="github.com/yourusername/GalPop.jl",
    devbranch="main",
)