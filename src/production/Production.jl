# Production features: building f from a ramp, chunk-file diagnostics. Depends on the bslLD core;
# the core does not refer to this module. Nothing is exported: import what the script uses, e.g.
#   using bslLD.Production: Diagnostic, write!, close!, gaussian_ramp, deposit!
module Production

using Serialization
using SpecialFunctions: erf
import PlasmaCore
import ..bslLD

include("initialization.jl")
include("diagnostics.jl")

end # module Production
