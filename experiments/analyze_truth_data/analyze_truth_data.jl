#=

    We create tons of .arrow files and save them in MLCondensateDistributions/data/processed

    We'd like to study these files to understand the data better, and to check for any issues before we train on them.


    Things to understand / check:

     - How does TKE affect the covariances?
     - Do we see different relationship between liq and ice and the other variables?

     - How does horizontal resolution affect the data?
     - How does vertical resolution affect the covariances? This might have a strong impact since we are 100% not homogeneous in the vertical
        - How do local gradients affect the data (we haven't saved gradients yet though)

     - How do liquid fraction and ice fraction affect the data? Do we see a strong relationship between these and the covariances?


     and more...


     Ideally for each of these questions we'd have a nice little routine that calculates different things and makes a nice plot that clearly shows what's going on.


     It would be good to create scorecards for:
        NN (all relevant variants)
        SHOC (need to find paper and create implementation)
        ADGC1 (need to find paper and create implementation -- i think this is prognostic tho so idk)
        ADGC2 (need to find paper and create implementation -- i think this is prognostic tho so idk)
        correlations implied in downgradient diffusion (we would need to save gradients to the arrow files and calculate eddy diffusivities which idk if we have data for....)
        etc (any other methods we can think of)


    analytic closures implemented in MLCondensateDistributions/experiments/analyze_truth_data/analytical_closures.jl
     

    Figures are saved to: MLCondensateDistributions/experiments/analyze_truth_data/figures

=#




