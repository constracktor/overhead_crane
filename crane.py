# imports for simulation
import numpy.matlib
import numpy as np
from scipy.integrate import solve_ivp
# imports for keyboard input
# use 'pip install pygame' in IPython console to load module
import pygame
import time
import sys
from pygame.locals import *
# imports for visualization
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
from mpl_toolkits.mplot3d import Axes3D

##############################################################################
# Functions
# Crane equations - Return values only for the discrete times defined in T_start_end
def crane(t,
          states_q,
          T_start_end,
          input_U):
    # System generalized coordinates q
    z, x, l, phi, theta, dz_dt, dx_dt, dl_dt, dphi_dt, dtheta_dt = states_q
    # System input forces
    # print(input_U)
    Ub = np.array([input_U[0], input_U[3]])
    Ut = np.array([input_U[1], input_U[4]])
    Ul = np.array([input_U[2], input_U[5]])
    # Base parameters
    g = 9.81  # m/s^2
    mc = 0.85  # kg
    mt = 5    # kg
    mb = 7    # kg
    ml = 2    # kg
    bt = 20   # Nm/s
    bb = 30   # Nm/s
    br = 50   # Nm/s

    # Inertia matrix
    M = np.zeros((5, 5))
    M[0, 0] = mt + mb + mc
    # M[0,1] = 0
    M[0, 2] = mc * np.sin(phi) * np.cos(theta)
    M[0, 3] = mc * l * np.cos(phi) * np.cos(theta)
    M[0, 4] = -mc * l * np.sin(phi) * np.sin(theta)

    # M[1,0] = 0
    M[1, 1] = mt + mc
    M[1, 2] = mc * np.sin(theta)
    # M[1,3]= 0
    M[1, 4] = mc * l * np.cos(theta)

    M[2, 0] = M[0, 2]
    M[2, 1] = M[1, 2]
    M[2, 2] = ml + mc
    # M[2,3] = 0
    # M[2,4] = 0

    M[3, 0] = M[0, 3]
    # M[3,1] = 0
    # M[3,2] = 0
    M[3, 3] = mc * l**2 * np.cos(theta)**2
    # M[3,4] = 0

    M[4, 0] = M[0, 4]
    M[4, 1] = M[1, 4]
    # M[4,2] = 0
    # M[4,3] = 0
    M[4, 4] = mc * l**2

    # Coriolis + Centripetal matrix
    C = np.zeros((5, 5))
    C[0, 0] = bb
    # C[0,1] = 0
    C[0, 2] = mc * np.cos(phi) * np.cos(theta) * dphi_dt \
        - mc * np.sin(phi) * np.sin(theta) * dtheta_dt
    C[0, 3] = mc * np.cos(phi) * np.cos(theta) * dl_dt \
        - mc * l * np.cos(phi) * np.sin(theta) * dtheta_dt \
        - mc * l * np.sin(phi) * np.cos(theta) * dphi_dt
    C[0, 4] = -mc * l * np.cos(phi) * np.sin(theta) * dphi_dt \
        - mc * np.sin(phi) * np.sin(theta) * dl_dt \
        - mc * l * np.sin(phi) * np.cos(theta) * dtheta_dt

    # C[1,0] = 0
    C[1, 1] = bt
    C[1, 2] = mc * np.cos(theta) * dtheta_dt
    # C[1,3]= 0
    C[1, 4] = mc * np.cos(theta) * dl_dt - mc * l * np.sin(theta) * dtheta_dt

    # C[2,0] = 0
    # C[2,1] = 0
    C[2, 2] = br
    C[2, 3] = -mc * l * np.cos(theta)**2 * dphi_dt
    C[2, 4] = -mc * l * dtheta_dt

    # C[3,0] = 0
    # C[3,1] = 0
    C[3, 2] = mc * l * np.cos(theta)**2 * dphi_dt
    C[3, 3] = mc * l * np.cos(theta)**2 * dl_dt \
        - mc * l**2 * np.cos(theta) * np.sin(theta) * dtheta_dt
    C[3, 4] = -mc * l**2 * np.cos(theta) * np.sin(theta) * dphi_dt

    # C[4,0] = 0
    # C[4,1] = 0
    C[4, 2] = mc * l * dtheta_dt
    C[4, 3] = mc * l**2 * np.cos(theta) * np.sin(theta) * dphi_dt
    C[4, 4] = mc * l * dl_dt

    # Gravitational matrix
    G = np.zeros(5)
    # G[0] = 0
    # G[1] = 0
    G[2] = -mc * g * np.cos(phi) * np.cos(theta)
    G[3] = mc * g * l * np.sin(phi) * np.cos(theta)
    G[4] = mc * g * l * np.cos(phi) * np.sin(theta)

    # Forces
    F = np.zeros(5)
    F[0] = np.interp(t, T_start_end, Ub)
    F[1] = np.interp(t, T_start_end, Ut)
    # G[2] is the gravity compensation
    F[2] = np.interp(t, T_start_end, Ul) + G[2]
    # F[3] = 0
    # F[4] = 0

    # Compute states_q_dt: Solve M * dq_dt_dt + C * dq_dt + G = F
    # as states_q = [q, dq_dt] reuse values
    dq_dt = states_q[5:10]
    # compute derivate of dq_dt: M * dq_dt_dt = ( F - G ) - C * dq_dt
    M_inverse = np.linalg.inv(M)
    dq_dt_dt = M_inverse.dot(F - G) - np.matmul(M_inverse, C).dot(dq_dt)
    # concatenate arrays
    states_q_dt = np.concatenate([dq_dt, dq_dt_dt])
    return states_q_dt


def boundary_treatment(states_q, z_limits, x_limits, l_limits):
    if states_q[0] > z_limits[1]:
        # z = bc
        states_q[0] = z_limits[1]
        # dz_dt = 0
        states_q[5] = 0
    elif states_q[0] < z_limits[0]:
        # z = bc
        states_q[0] = z_limits[0]
        # dz_dt = 0
        states_q[5] = 0
    if states_q[1] > x_limits[1]:
        # x = bc
        states_q[1] = x_limits[1]
        # dx_dt = 0
        states_q[6] = 0
    elif states_q[1] < x_limits[0]:
        # x = bc
        states_q[1] = x_limits[0]
        # dx_dt = 0
        states_q[6] = 0
    if states_q[2] > l_limits[1]:
        # l = bc
        states_q[2] = l_limits[1]
        # dl_dt = 0
        states_q[7] = 0
    elif states_q[2] < l_limits[0]:
        # l = bc
        states_q[2] = l_limits[0]
        # dl_dt = 0
        states_q[7] = 0
    # Return potentially modified states
    return states_q

# Get input from Console
def key_press_callback(DELTA_F1, DELTA_F2, DELTA_F3, screen):
    # define applied force
    applied_force = 10
    delta_F1 = DELTA_F1
    delta_F2 = DELTA_F2
    delta_F3 = DELTA_F3
    for event in pygame.event.get():
        # when window is closed temrinate
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        # resize window
        if event.type == VIDEORESIZE:
            screen = pygame.display.set_mode(
                event.size, HWSURFACE | DOUBLEBUF | RESIZABLE)
        # when key is pressed force is applied
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                delta_F1 = -applied_force
            if event.key == pygame.K_RIGHT:
                delta_F1 = applied_force
            if event.key == pygame.K_DOWN:
                delta_F2 = -applied_force
            if event.key == pygame.K_UP:
                delta_F2 = applied_force
            if event.key == pygame.K_q:
                delta_F3 = -applied_force
            if event.key == pygame.K_a:
                delta_F3 = applied_force
            if event.key == pygame.K_x:
                pygame.quit()
                sys.exit()
        # when key is released force becomes 0
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                delta_F1 = 0
            if event.key == pygame.K_RIGHT:
                delta_F1 = 0
            if event.key == pygame.K_DOWN:
                delta_F2 = 0
            if event.key == pygame.K_UP:
                delta_F2 = 0
            if event.key == pygame.K_q:
                delta_F3 = 0
            if event.key == pygame.K_a:
                delta_F3 = 0
    return delta_F1, delta_F2, delta_F3, screen

# Plots the current state and converts the image for PyGame
def plot_crane(states_q, z_limits, x_limits, l_limits, screen, fake_screen):
    # Create Figure
    fig = plt.figure(figsize=[5, 5], dpi=200)
    ax = plt.axes(projection='3d')
    # Set and label axes
    ax.set_ylim3d([-0.35, 0.35])
    ax.set_xlim3d([-0.35, 0.35])
    ax.set_ylabel('$x$-direction')
    ax.set_xlabel('$z$-direction')
    ax.set_zlabel('$y$-direction')
    # Draw boundary lines
    ax.plot([z_limits[0], z_limits[0]], [x_limits[0], x_limits[0]],
            zs=[0, -l_limits[1]], color='yellow', linestyle='dashed')
    ax.plot([z_limits[0], z_limits[0]], [x_limits[1], x_limits[1]],
            zs=[0, -l_limits[1]], color='yellow', linestyle='dashed')
    ax.plot([z_limits[0], z_limits[0]], [x_limits[0], x_limits[1]],
            zs=[0, 0], color='yellow', linestyle='dashed')
    ax.plot([z_limits[0], z_limits[0]], [x_limits[0], x_limits[1]],
            zs=[-l_limits[1], -l_limits[1]], color='yellow', linestyle='dashed')
    ax.plot([z_limits[0], z_limits[1]], [x_limits[0], x_limits[0]],
            zs=[0, 0], color='yellow', linestyle='dashed')
    ax.plot([z_limits[0], z_limits[1]], [x_limits[0], x_limits[0]],
            zs=[-l_limits[1], -l_limits[1]], color='yellow', linestyle='dashed')
    ax.plot([z_limits[0], z_limits[1]], [x_limits[1], x_limits[1]],
            zs=[0, 0], color='yellow', linestyle='dashed')
    ax.plot([z_limits[0], z_limits[1]], [x_limits[1], x_limits[1]],
            zs=[-l_limits[1], -l_limits[1]], color='yellow', linestyle='dashed')
    ax.plot([z_limits[1], z_limits[1]], [x_limits[0], x_limits[0]],
            zs=[0, -l_limits[1]], color='yellow', linestyle='dashed')
    ax.plot([z_limits[1], z_limits[1]], [x_limits[1], x_limits[1]],
            zs=[0, -l_limits[1]], color='yellow', linestyle='dashed')
    ax.plot([z_limits[1], z_limits[1]], [x_limits[0], x_limits[1]],
            zs=[0, 0], color='yellow', linestyle='dashed')
    ax.plot([z_limits[1], z_limits[1]], [x_limits[0], x_limits[1]],
            zs=[-l_limits[1], -l_limits[1]], color='yellow', linestyle='dashed')
    # Plot real values
    l_pendulum = states_q[2]
    # Angles
    phi = states_q[3]
    theta = states_q[4]
    # Points crane
    z_crane = states_q[0]
    x_crane = states_q[1]
    y_crane = 0
    # Points load
    z_load = z_crane + l_pendulum * np.sin(phi)
    x_load = x_crane + l_pendulum * np.sin(theta)
    y_load = -l_pendulum * np.cos(phi) * np.cos(theta)
    # Plot crane
    scatter_crane = ax.scatter([z_crane], [x_crane], [y_crane],
                               color='red', linewidth=1)
    # Plot load
    scatter_load = ax.scatter([z_load], [x_load], [y_load],
                              color='red', linewidth=10)
    # Draw pendulum
    line_pendulum, = ax.plot([z_crane, z_load], [x_crane, x_load],
                             [y_crane, y_load], color='blue')
    # Render as canvas
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    plt.close(fig)
    # Convert to raw data to use in pygame
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()
    # Erase screen
    fake_screen.fill((255, 255, 255))
    # add new image to PyGame
    surf = pygame.image.fromstring(raw_data, size, "RGB")
    # Show new image in game window

    fake_screen.blit(surf, (0, 0))
    screen.blit(pygame.transform.scale(
        fake_screen, screen.get_rect().size), (0, 0))

    pygame.display.update()

# Data Generator for Gaussian Process Application
# def simulate_crane_gp(initial_states_q,
#                       input_U,
#                       noise_w,
#                       timesteps_TT,
#                       z_limits,
#                       x_limits,
#                       l_limits):
#     # Initialize state and output matrix
#     state_matrix_X = initial_states_q.reshape(1, 10)
#     output_matrix_Y = initial_states_q[0:5].reshape(1, 5) \
#         + noise_w * np.random.rand(1, 5)
#     # Solver parameters
#     number_timesteps = 10
#     for index in range(0, timesteps_TT.size - 1):
#         # Get current time step  and input
#         current_intervall_T = timesteps_TT[index:(index + 2)]
#         current_input_U = input_U[index:(index + 2), :].reshape(1, 6)[0, :]
#         timesteps_solver = np.linspace(current_intervall_T[0],
#                                        current_intervall_T[1], number_timesteps)
#         # Parameters of the system
#         p = (current_intervall_T, current_input_U)
#         # Solve differential equation numerically
#         results = solve_ivp(crane,
#                             current_intervall_T,
#                             initial_states_q,
#                             args=p,
#                             t_eval=timesteps_solver)
#         # Reshape the result to numpy array
#         new_state_vector_X = results.y[:, -1].reshape(1, -1)
#         # Compute noisy output vector
#         new_output_vector_Y = new_state_vector_X[0,
#                                                  0:5] + noise_w * np.random.randn(1, 5)
#         # Update matrices
#         state_matrix_X = np.vstack((state_matrix_X, new_state_vector_X))
#         output_matrix_Y = np.vstack((output_matrix_Y, new_output_vector_Y))
#         # Update states
#         initial_states_q = new_state_vector_X[0, :]
#         # Check boundary conditions bc
#         initial_states_q = boundary_treatment(initial_states_q,
#                                               z_limits,
#                                               x_limits,
#                                               l_limits)
#     return state_matrix_X, output_matrix_Y

# While true loop of the crane model which PyGame visualization and interactive keyboard input
def simulate_crane_loop():
    # Simulation Parameters
    # Time discretization
    dt = 0.1
    # Initial Condition
    # starting from rest
    # z, x, l, phi, theta, dz_dt, dx_dt, dl_dt, dphi_dt, dtheta_dt
    initial_states_q = np.array([0, 0, 0.3, 0, 0, 0, 0, 0, 0, 0])
    # Force difference to previous step
    DELTA_F1 = 0
    DELTA_F2 = 0
    DELTA_F3 = 0
    # Boundary Conditions
    z_limits = np.array([-0.3, 0.3])
    x_limits = np.array([-0.3, 0.3])
    l_limits = np.array([0.1, 0.4])
    # Output noise
    noise_w = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
    intervall_T = np.array([0, dt])
    input_U = np.array([0, 0, 0, 0, 0, 0])
    # Termination variable
    PROCEED = True
    # Force difference to previous step
    DELTA_F1 = 0
    DELTA_F2 = 0
    DELTA_F3 = 0
    # Initialize PyGame to allow Keyboard input
    screen = pygame.display.set_mode(
        (1000, 1000), HWSURFACE | DOUBLEBUF | RESIZABLE)
    fake_screen = screen.copy()
    pygame.init()
    clock = pygame.time.Clock()
    pygame.display.set_caption('Crane Simulator')
    # surf = plot_crane(initial_states_q, z_limits, x_limits, l_limits, screen)
    pygame.mouse.set_visible(1)

    # while-loop until window closed or 'x' pressed
    while True:
        ##########################################################################
        # Keyboard Part
        # set terminal variable and forces according to keyboard input
        DELTA_F1, DELTA_F2, DELTA_F3, screen = key_press_callback(DELTA_F1,
                                                                  DELTA_F2,
                                                                  DELTA_F3,
                                                                  screen)
        ##########################################################################
        # Simulation Part
        input_U = np.array([0, 0, 0, DELTA_F1, DELTA_F2, DELTA_F3])
        # Run simulation
        number_timesteps = 10
        timesteps_solver = np.linspace(intervall_T[0], intervall_T[1],
                                       number_timesteps)
        # Parameters of the system
        p = (intervall_T, input_U)
        # Solve differential equation numerically
        results = solve_ivp(crane,
                            intervall_T,
                            initial_states_q,
                            args=p,
                            t_eval=timesteps_solver)
        # Reshape the output to numpy array
        state_matrix_X = np.hstack([i.reshape(-1, 1) for i in results.y])
        # Compute noisy output
        # output_vector_Y = X[:,0:5] + noise_w * np.random.rand(number_timesteps,5)
        # Update time
        intervall_T = intervall_T + dt
        # Update states
        initial_states_q = state_matrix_X[-1, :]
        # Check boundary conditions bc
        initial_states_q = boundary_treatment(initial_states_q,
                                              z_limits,
                                              x_limits,
                                              l_limits)
        ##########################################################################
        # Animation Part
        plot_crane(initial_states_q,
                   z_limits,
                   x_limits,
                   l_limits,
                   screen,
                   fake_screen)


##############################################################################
# Run Crane Simulation with visualization in PyGame
simulate_crane_loop()
