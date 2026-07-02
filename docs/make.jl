using Documenter
using JSON3

# ── 0. Clean up generated files from previous builds ─────────────────────────
for stale in ("src/index.md", "src/examples.md")
    f = joinpath(@__DIR__, stale)
    isfile(f) && rm(f)
end

# ── 1. Auto-discover notebooks ────────────────────────────────────────────────
examples_dir = joinpath(@__DIR__, "..", "examples")
notebooks = sort(filter(f -> endswith(f, ".ipynb"), readdir(examples_dir, join = true)))

# ── 2. Run nbconvert WITH execution; output goes into docs/src/notebooks/ ─────
#    Placed here (before makedocs) so Documenter copies them as static assets.
#    Pass --reconvert to force re-conversion even if HTML already exists.
notebooks_src = joinpath(@__DIR__, "src", "notebooks")
mkpath(notebooks_src)
nbconvert = get(ENV, "NBCONVERT", "jupyter-nbconvert")
force_reconvert = "--reconvert" in ARGS
for nb in notebooks
    name = splitext(basename(nb))[1]
    out_html = joinpath(notebooks_src, name * ".html")
    if !force_reconvert && isfile(out_html)
        @info "Skipping $name (already converted; use --reconvert to force)"
        continue
    end
    @info "Converting (executing) $name"
    try
        run(`$(nbconvert) --to html --execute --no-input
            --ExecutePreprocessor.timeout=600
            --output-dir $(notebooks_src)
            $(nb)`)
    catch e
        @warn "nbconvert failed for $name: $e"
    end
end

# ── 3. Extract first non-heading line from the first markdown cell ─────────────
function first_description(nb_path)
    nb = JSON3.read(read(nb_path, String))
    for cell in nb["cells"]
        cell["cell_type"] == "markdown" || continue
        src = join(cell["source"])
        for line in split(src, "\n")
            stripped = strip(lstrip(line, '#'))
            stripped = strip(stripped)
            isempty(stripped) && continue
            startswith(stripped, "\$\$") && continue
            return string(stripped)
        end
    end
    return replace(splitext(basename(nb_path))[1], "_" => " ")
end

# ── 4. README + examples table → index.md (single source of truth) ───────────
cp(
    joinpath(@__DIR__, "..", "README.md"),
    joinpath(@__DIR__, "src", "index.md"),
    force = true,
)

open(joinpath(@__DIR__, "src", "index.md"), "a") do io
    println(io, "\n---\n\n## Examples\n")
    println(io, "| Notebook | Description |")
    println(io, "|----------|-------------|")
    for nb in notebooks
        name = splitext(basename(nb))[1]
        isfile(joinpath(notebooks_src, name * ".html")) || continue
        label = replace(name, "_" => " ")
        desc = first_description(nb)
        # Link is relative from build/index.html → build/notebooks/X.html
        println(io, "| [$(label)](notebooks/$(name).html) | $(desc) |")
    end
end

# ── 5. makedocs ───────────────────────────────────────────────────────────────
makedocs(
    sitename = "bslLD",
    authors = "Mario Raeth",
    pages = ["Home" => "index.md"],
    format = Documenter.HTML(
        prettyurls = true,
        canonical = "https://mraeth.github.io/bslLD",
    ),
    doctest = false,
)

# ── 6. Deploy ─────────────────────────────────────────────────────────────────
deploydocs(repo = "github.com/mraeth/bslLD.git", devbranch = "main", push_preview = false)
