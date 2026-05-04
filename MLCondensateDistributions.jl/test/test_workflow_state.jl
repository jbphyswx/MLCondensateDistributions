using Test: Test

include("../utils/workflow_state.jl")
using .WorkflowState: WorkflowState

Test.@testset "Workflow state manifest" begin
    temp_path = joinpath(@__DIR__, "workflow_state_test.tsv")
    if isfile(temp_path)
        rm(temp_path; force=true)
    end

    WorkflowState.record_checked_case!(temp_path, 10, 1, "amip"; status="cloudfree", rows=0, timesteps=73)
    records = WorkflowState.load_checked_cases(temp_path)

    key = WorkflowState.case_key(10, 1, "amip")
    Test.@test haskey(records, key)
    Test.@test records[key].status == "cloudfree"
    Test.@test records[key].timesteps == 73
    Test.@test WorkflowState.was_checked(temp_path, key)

    rm(temp_path; force=true)
end