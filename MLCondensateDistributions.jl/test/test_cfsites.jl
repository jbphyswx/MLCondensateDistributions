using Test: Test
using DimensionalData: DimensionalData as DD
using NCDatasets: NCDatasets as NC

# Include the cfSites module from utils
include("../utils/cfSites.jl")
using .cfSites: cfSites

Test.@testset "cfSites 4D Data Retrieval Tests" begin
    # Testing configuration that exists on the HPC
    cfSite    = 10
    model     = "CNRM-CM5"
    month     = 1
    exp       = "amip"
    
    les_dir = cfSites.get_cfSite_les_dir(cfSite, forcing_model=model, month=month, experiment=exp)
    
    Test.@test ispath(les_dir)
    
    @info "Loading 'qt' and 'temperature'. Since this targets the active HPC filesystem, it might take a moment..."
    data = cfSites.load_4d_fields(les_dir, ["qt", "temperature"])
    
    # 1. Check returned structure
    Test.@test data isa DD.DimStack
    Test.@test haskey(data, :qt)
    Test.@test haskey(data, :temperature)
    Test.@test haskey(data, :p0)
    
    # 2. Check types
    Test.@test data.qt isa DD.DimArray
    Test.@test data.temperature isa DD.DimArray
    Test.@test data.p0 isa DD.DimArray
    
    # 3. Check dimensionality
    Test.@test length(DD.dims(data.qt)) == 4           # X, Y, Z, Ti
    Test.@test length(DD.dims(data.temperature)) == 4  
    Test.@test length(DD.dims(data.p0)) == 1           # Z mapping
    
    @info "Saving loaded data to 'test/test_output.nc' for inspection."
    output_path = joinpath(dirname(@__FILE__), "test_output.nc")
    
    NC.NCDataset(output_path, "c") do ds
        # Define shared dimensions from the stack
        for (i, d) in enumerate(DD.dims(data.qt))
            dim_name = string(DD.name(d))
            NC.defDim(ds, dim_name, length(d))
            NC.defVar(ds, dim_name, parent(d), (dim_name,))
        end
        
        # Define and save variables
        for vname in keys(data)
            v_array = data[vname]
            v_dims = Tuple(string(DD.name(d)) for d in DD.dims(v_array))
            NC.defVar(ds, string(vname), parent(v_array), v_dims)
        end
    end
    
    @info "Completed DimensionalData mapping test successfully!"
end
