import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np


# takes multiple simulations of the same number of steps and makes them more "plottable"
def preprocess_profiles(model_profiles,maxVelocityIterations, maxPositionIterations):
    results = model_profiles
    output = {}
    output["contactsSolved"] = np.mean([res["contactsSolved"] for res in results],
                                               axis=0, dtype=np.float64)
    output["stepTimes"] = np.mean([res["totalStepTimes"] for res in results],
                                  axis=0, dtype=np.float64)

    output["totalVelocityIterations"] = np.mean([res["totalVelocityIterations"] for res in results],
                                                axis=0, dtype=np.float64)

    output["totalPositionIterations"] = np.mean([res["totalPositionIterations"] for res in results],
                                                axis=0, dtype=np.float64)
    # We filter out the convergence rates for steps with no contacts
    velocityLambdaInfLists = [list(filter(lambda l: l[-1] != 0, res["velocityLambdaInfNorms"])) for res in results]
    velocityLambdaTwoLists = [list(filter(lambda l: l[-1] != 0, res["velocityLambdaTwoNorms"])) for res in results]

    # Pad convergence rates to be same length, which is the max, by repeating the last element
    paddedVelocityLambdaInfLists = [[l + [l[-1]]*(maxVelocityIterations-len(l)) for l in lambdas]
                                     for lambdas in velocityLambdaInfLists]
    paddedVelocityLambdaTwoLists = [[l + [l[-1]]*(maxVelocityIterations-len(l)) for l in lambdas]
                                     for lambdas in velocityLambdaTwoLists]

    # We transform the data into an array
    velocityLambdaInfArray = np.concatenate([np.array(l) for l in paddedVelocityLambdaInfLists])
    velocityLambdaTwoArray = np.concatenate([np.array(l) for l in paddedVelocityLambdaTwoLists])

    output["velocityLambdasInf"] = velocityLambdaInfArray
    output["velocityLambdasTwo"] = velocityLambdaTwoArray


    # We determine the number of steps still iterating for each iteration
    velocityIteratorInfCountLists = [[np.sum([len(l) >= i for l in lambdas])
                                   for i in range(maxVelocityIterations)]
                                  for lambdas in velocityLambdaInfLists]
    velocityIteratorTwoCountLists = [[np.sum([len(l) >= i for l in lambdas])
                                   for i in range(maxVelocityIterations)]
                                  for lambdas in velocityLambdaTwoLists]

    # Pad iterator counts to be same length, by adding zeros
    paddedVelocityIteratorInfCountLists = [l + [0]*(maxVelocityIterations-len(l))
                                        for l in velocityIteratorInfCountLists]
    paddedVelocityIteratorTwoCountLists = [l + [0]*(maxVelocityIterations-len(l))
                                        for l in velocityIteratorTwoCountLists]

    # We take the mean of the iterator count lists
    output["velocityIteratorInfCounts"] = np.mean(paddedVelocityIteratorInfCountLists, axis=0, dtype=np.float64)
    output["velocityIteratorTwoCounts"] = np.mean(paddedVelocityIteratorTwoCountLists, axis=0, dtype=np.float64)
    
    # We filter out the convergence rates for steps with no contacts
    positionLambdaLists = [list(filter(lambda l: l[-1] != 0, res["positionLambdas"])) for res in results]

    # Pad convergence rates to be same length, which is the max, by repeating the last element
    paddedPositionLambdaLists = [[l + [l[-1]]*(maxPositionIterations-len(l)) for l in lambdas]
                                 for lambdas in positionLambdaLists]

    # We transform the data into an array
    positionLambdaArray = np.concatenate([np.array(l) for l in paddedPositionLambdaLists])
    output["positionLambdas"] = positionLambdaArray

    # We determine the number of steps still iterating for each iteration
    positionIteratorCountLists = [[np.sum([len(l) >= i for l in lambdas])
                                   for i in range(maxPositionIterations)]
                                  for lambdas in positionLambdaLists]

    # Pad iterator counts to be same length, by adding zeros
    paddedPositionIteratorCountLists = [l + [0]*(maxPositionIterations-len(l))
                                        for l in positionIteratorCountLists]

    # We take the mean of the iterator count lists
    output["positionIteratorCounts"] = np.mean(paddedPositionIteratorCountLists, axis = 0, dtype=np.float64)
    
    normalPairsListLists = [res["normalPairs"] for res in results]
    tangentPairsListLists = [res["tangentPairs"] for res in results]

    normalMSELists = [[np.mean([(pair[0] - pair[1])**2 for pair in step]) for step in world]
                      for world in normalPairsListLists]
    tangentMSELists = [[np.mean([(pair[0] - pair[1])**2 for pair in step]) for step in world]
                       for world in tangentPairsListLists]

    normalErrors = np.nanmean(normalMSELists, axis=0)
    tangentErrors = np.nanmean(tangentMSELists, axis=0)

    output["normalErrors"] = normalErrors
    output["tangentErrors"] = tangentErrors
    
    return output

def show_iterations_plot(to_plot,colors,smoothingFactor=1,skip_plots=[]):
    names = list(to_plot.keys())
    processed_results = list(to_plot.values())
    fig = plt.figure("Iterations", figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    fig.suptitle("Iteration Counters")

    # Times
    ax1 = fig.add_subplot(221)
    for i in range(len(processed_results)):
        if names[i] in skip_plots: continue
        ax1.plot(
            smooth(processed_results[i]["stepTimes"],  smoothingFactor),
            ls="solid", c=colors[names[i]], label=names[i]
        )
    ax1.legend()
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Step time")
    ax1.set_title("Time taken for each step")

    # Contacts
    ax1 = fig.add_subplot(222)
    for i in range(len(processed_results)):
        if names[i] in skip_plots: continue
        ax1.plot(
            processed_results[i]["contactsSolved"],
            ls="solid", c=colors[names[i]], label=names[i]
        )
    ax1.legend()
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Total number of contacts")
    ax1.set_title("Contact numbers for each step")

    # Velocity iterations
    ax1 = fig.add_subplot(223)
    for i in range(len(processed_results)):
        if names[i] in skip_plots: continue
        ax1.plot(
            smooth(processed_results[i]["totalVelocityIterations"], smoothingFactor),
            ls="solid", c=colors[names[i]], label=names[i]
        )
    ax1.legend()
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Total number of iterations")
    ax1.set_title("Velocity iterations numbers")
    
    # Position iterations
    ax1 = fig.add_subplot(224)
    for i in range(len(processed_results)):
        if names[i] in skip_plots: continue
        ax1.plot(
            smooth(processed_results[i]["totalPositionIterations"], smoothingFactor),
            ls="solid", c=colors[names[i]], label=names[i]
        )
    ax1.legend()
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Total number of iterations")
    ax1.set_title("Position iterations numbers")

def show_velocity_convergence(to_plot,colors,velocityThreshold,skip_plots=[],
                            velocityPercentile=50,velocityCutoff = 500, velocityAddQuantiles=False):
    names = list(to_plot.keys())
    processed_results = list(to_plot.values())
    
    # Percentile to plot if set
    # 0 for min, 25 for 1st quantile, 50 for median, 75 for 3rd quantile, 100 for max
    velocityConvDataInf = [np.percentile(res["velocityLambdasInf"], velocityPercentile, axis=0)
                        for res in processed_results]
    velocityQ1Inf = [np.percentile(res["velocityLambdasInf"], 25, axis=0)
                  for res in processed_results]
    velocityQ3Inf = [np.percentile(res["velocityLambdasInf"], 75, axis=0)
                  for res in processed_results]

    velocityConvDataMean = [np.mean(res["velocityLambdasTwo"], axis=0, dtype=np.float64)
                        for res in processed_results]

    velocityIteratorInfData = [res["velocityIteratorInfCounts"] for res in processed_results]
    velocityIteratorTwoData = [res["velocityIteratorTwoCounts"] for res in processed_results]


    fig = plt.figure("Velocity Convergence", figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    fig.suptitle("Velocity Convergence Rates")

    velocityYlim = [velocityThreshold/2, 10**1]


    # Full lambda convergence rates
    ax1 = fig.add_subplot(221)
    for i in range(len(processed_results)):
        if names[i] in skip_plots: continue
        ax1.semilogy(velocityConvDataInf[i], ls="solid", c=colors[names[i]], label=names[i])
        if velocityAddQuantiles:
            ax1.semilogy(velocityQ1Inf[i], ls="dashed", c=colors[names[i]])
            ax1.semilogy(velocityQ3Inf[i], ls="dashed", c=colors[names[i]])
    ax1.legend()
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Lambda Inf-Norm")
    ax1.set_title("Velocity Lambda Convergence Rate - All iterations, " + str(velocityPercentile) + "%")

    # Counters
    ax1 = fig.add_subplot(222)
    for i in range(len(processed_results)):
        if names[i] in skip_plots: continue
        ax1.plot(velocityIteratorInfData[i], ls="solid", c=colors[names[i]], label=names[i])
    ax1.legend()
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Remaining")
    ax1.set_title("Number of iterators left for each iteration")

    # Limited lambda inf-norm convergence rates
    ax1 = fig.add_subplot(223)
    for i in range(len(processed_results)):
        if names[i] in skip_plots: continue
        ax1.semilogy(velocityConvDataInf[i], ls="solid", c=colors[names[i]], label=names[i])
        if velocityAddQuantiles:
            ax1.semilogy(velocityQ1Inf[i], ls="dashed", c=colors[names[i]])
            ax1.semilogy(velocityQ3Inf[i], ls="dashed", c=colors[names[i]])
    ax1.set_xlim([0, velocityCutoff])
    ax1.set_ylim(velocityYlim)
    ax1.legend()
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Lambda Inf-Norm")
    ax1.set_title("Velocity Lambda Convergence Rate - Cutoff, " + str(velocityPercentile) + "%")

    # Limited counters
    ax1 = fig.add_subplot(224)
    for i in range(len(processed_results)):
        if names[i] in skip_plots: continue
        ax1.plot(velocityIteratorTwoData[i], ls="solid", c=colors[names[i]], label=names[i])
    ax1.set_xlim([0, velocityCutoff])
    ax1.legend()
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Remaining")
    ax1.set_title("Number of iterators left for each iteration")

def show_position_convergence(to_plot,colors,positionThreshold,skip_plots=[],
                    positionPercentile=50,positionCutoff=100, positionAddQuantiles=False):
    names = list(to_plot.keys())
    processed_results = list(to_plot.values())
    
    positionConvData = [np.percentile(res["positionLambdas"], positionPercentile, axis=0)
                        for res in processed_results]
    positionQ1 = [np.percentile(res["positionLambdas"], 25, axis=0)
                  for res in processed_results]
    positionQ3 = [np.percentile(res["positionLambdas"], 75, axis=0)
                  for res in processed_results]
    positionConvData = [np.mean(res["positionLambdas"], axis=0, dtype=np.float64)
                        for res in processed_results]

    # Iterator counts
    positionIteratorData = [res["positionIteratorCounts"] for res in processed_results]


    fig = plt.figure("Position Convergence", figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    fig.suptitle("Position Convergence Rates")

    positionYlim = [positionThreshold/2, 10**1]


    # Full lambda convergence rates
    ax1 = fig.add_subplot(221)
    for i in range(len(processed_results)):
        if names[i] in skip_plots: continue
        ax1.semilogy(positionConvData[i], ls="solid", c=colors[names[i]], label=names[i])
        if positionAddQuantiles:
            ax1.semilogy(positionQ1[i], ls="dashed", c=colors[names[i]])
            ax1.semilogy(positionQ3[i], ls="dashed", c=colors[names[i]])
    ax1.legend()
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Lambda")
    ax1.set_title("Position Lambda Convergence Rate - All iterations, " + str(positionPercentile) + "%")

    # Counters
    ax1 = fig.add_subplot(222)
    for i in range(len(processed_results)):
        if names[i] in skip_plots: continue
        ax1.plot(positionIteratorData[i], ls="solid", c=colors[names[i]], label=names[i])
    ax1.legend()
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Remaining")
    ax1.set_title("Number of iterators left for each iteration")

    # Limited lambda convergence rates
    ax1 = fig.add_subplot(223)
    for i in range(len(processed_results)):
        if names[i] in skip_plots: continue
        ax1.semilogy(positionConvData[i], ls="solid", c=colors[names[i]], label=names[i])
        if positionAddQuantiles:
            ax1.semilogy(positionQ1[i], ls="dashed", c=colors[names[i]])
            ax1.semilogy(positionQ3[i], ls="dashed", c=colors[names[i]])
    ax1.set_xlim([0, positionCutoff])
    ax1.set_ylim(positionYlim)
    ax1.legend()
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Lambda")
    ax1.set_title("Position Lambda Convergence Rate - Cutoff, " + str(positionPercentile) + "%")

    # Limited counters
    ax1 = fig.add_subplot(224)
    for i in range(len(processed_results)):
        if names[i] in skip_plots: continue
        ax1.plot(positionIteratorData[i], ls="solid", c=colors[names[i]], label=names[i])
    ax1.set_xlim([0, positionCutoff])
    ax1.legend()
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Remaining")
    ax1.set_title("Number of iterators left for each iteration")

def show_lambda_errors(to_plot,colors,skip_plots=[],errorSmoothingFactor=3):
    names = list(to_plot.keys())
    processed_results = list(to_plot.values())
    steps = processed_results[0]["normalErrors"].size
    normalErrors = [smooth(pro_res["normalErrors"], errorSmoothingFactor) for pro_res in processed_results]
    tangentErrors = [smooth(pro_res["tangentErrors"], errorSmoothingFactor) for pro_res in processed_results]

    normalMin = max(10**-5, np.nanmin([np.nanmin(l) for l in normalErrors]))
    normalMax = max([np.nanmax(l) for l in normalErrors])

    tangentMin = max(10**-5, np.nanmin([np.nanmin(l) for l in tangentErrors]))
    tangentMax = max([np.nanmax(l) for l in tangentErrors])

    # --- Lambda errors plots ---
    fig = plt.figure("Lambda Errors", figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
    fig.suptitle("Lambda Errors")

    # Normal Errors
    ax1 = fig.add_subplot(221)
    for i in range(len(processed_results)):
        if names[i] in skip_plots: continue
        ax1.semilogy(normalErrors[i], ls="solid", c=colors[names[i]], label=names[i])
    ax1.set_xlim([0, steps])
    ax1.set_ylim([normalMin-1e-5, normalMax+1000])
    ax1.legend()
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Error")
    ax1.set_title("Normal Impulse Errors")

    # Tangent Errors
    ax1 = fig.add_subplot(222)
    for i in range(len(processed_results)):
        if names[i] in skip_plots: continue
        ax1.semilogy(tangentErrors[i], ls="solid", c=colors[names[i]], label=names[i])
    ax1.set_xlim([0, steps])
    ax1.set_ylim([tangentMin-1e-5, tangentMax+1000])
    ax1.legend()
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Error")
    ax1.set_title("Tangent Impulse Errors")

    # Contacts
    ax1 = fig.add_subplot(223)
    for i in range(len(processed_results)):
        if names[i] in skip_plots: continue
        ax1.plot(processed_results[i]["contactsSolved"], ls="solid", c=colors[names[i]], label=names[i])
    ax1.legend()
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Total number of contacts")
    ax1.set_title("Contact numbers for each step")


# Function for smoothing data using a moving average as far as I recall
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
