import numpy as np
import matplotlib.pyplot as plt

# simulated ground truth and noisy measurements
np.random.seed(42)

N = 50 # Number of steps
true_position = np.linspace(0,10,N) 
velocity = (true_position[1]-true_position[0])
measurements = true_position + np.random.normal(0,0.5,N)

# Kalman Filter parameters
x = 0.0                      # Initial estimate of position
P = 1.0                      # Initial estimate uncertainty
Q = 0.01                     # Process noise (small, system is smooth)
R = 0.5**2                   # Measurement noise (based on sensor)

estimates = []

for z in measurements:
    # Prediction Steps (Motion Model)
    x = x + velocity
    P = P + Q

    # Update Step (Measurement Model)
    K = P/(P+R) # Kalman Gain
    x = x + K*(z-x) # Update estimate
    P = (1-K)*P # Update uncertainty

    estimates.append(x)

# Plotting
plt.plot(true_position, label="True Position")
plt.plot(measurements, label="Measurements", linestyle='dotted')
plt.plot(estimates, label="KF Estimate")
plt.legend()
plt.title("1D Kalman Filter")
plt.xlabel("Time Step")
plt.ylabel("Position")
plt.show()
