
import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate):
    dx = 2.0 * math.cos(math.radians(45))
    dy = 2.0 * math.sin(math.radians(45))
    dTheta = math.radians(90)
    relative_pose = gtsam.Pose2(dx, dy, dTheta)
    # TODO: Add the odometry factor between X(3) and X(4) to the graph (BetweenFactorPose2)
    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), relative_pose, ODOMETRY_NOISE))

    # TODO: Based on the odometry, find the initial estimate for the pose of X(4) and add it to the graph
    if initial_estimate.exists(X(3)):
        pose_3 = initial_estimate.atPose2(X(3))
        initial_guess_4 = pose_3.compose(relative_pose)
        initial_estimate.insert(X(4), initial_guess_4)
    return graph, initial_estimate