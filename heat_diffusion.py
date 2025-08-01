import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from concurrent.futures import ProcessPoolExecutor, as_completed

class ThermalSource:
    def __init__(self,
                 source_type="none",
                 A=50,
                 center=(0.5, 0.5),
                 sigma=0.02,
                 R=0.3,
                 omega=2 * np.pi / 50,
                 dt = 1e-3,
                 t_pulse=20):
        self.source_type = source_type
        self.A = A
        self.center = center
        self.sigma = sigma
        self.R = R
        self.omega = omega
        self.t_pulse = t_pulse
        self.dt = dt

    def stationary_gaussian(self, X, Y):
        x0, y0 = self.center
        return self.A * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * self.sigma**2))

    def lateral_gaussian(self, X, Y, t):
        x0, y0 = self.center
        x_t = x0 + self.R * np.sin(self.omega * t)
        return self.A * np.exp(-((X - x_t)**2 + (Y - y0)**2) / (2 * self.sigma**2))

    def circular_gaussian(self, X, Y, t):
        x0, y0 = self.center
        x_t = x0 + self.R * np.sin(self.omega * t)
        y_t = y0 + self.R * np.cos(self.omega * t)
        return self.A * np.exp(-((X - x_t)**2 + (Y - y_t)**2) / (2 * self.sigma**2))

    def uniform(self, X, Y):
        return self.A * np.ones_like(X)

    def pulsed(self, X, Y, t):
        step = int(t / self.dt)
        if (step % self.t_pulse) < self.t_pulse // 2:
            x0, y0 = self.center
            return self.A * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * self.sigma ** 2))
        else:
            return np.zeros_like(X)

    def get_value(self, X, Y, t):
        """
        Returns the thermal source function to be used based on the source selected by the user
        :return: Function
        """
        method_map = {
            "stationary_gaussian": lambda: self.stationary_gaussian(X, Y),
            "lateral_gaussian": lambda: self.lateral_gaussian(X, Y, t),
            "circular_gaussian": lambda: self.circular_gaussian(X, Y, t),
            "uniform": lambda: self.uniform(X, Y),
            "pulsed": lambda: self.pulsed(X, Y, t),
        }
        return method_map.get(self.source_type, lambda: np.zeros_like(X))()

class Grid:
    def __init__(self, width, height, step, dt, nt):
        self.width = width
        self.height = height
        self.dx = step
        self.dy = step
        self.dt = dt
        self.nt = nt
        self.nx = int(width / step)
        self.ny = int(height / step)
        x = np.linspace(0, self.width, self.nx)
        self.temp_matrix_old = np.zeros((self.nx, self.ny))
        self.temp_matrix_new = np.zeros((self.nx, self.ny))
        y = np.linspace(0, self.height, self.ny)
        self.X, self.Y = np.meshgrid(x, y)
        self.sources = []
        self.alpha_matrix = None
        self.rho_matrix = None
        self.c_matrix = None
        self.alpha_x_matrix = None
        self.alpha_y_matrix = None
        self.kx_matrix = None
        self.ky_matrix = None

    def set_initial_temperature(self, temperature):
        """
        Sets the initial temperature of the grid
        """
        self.temp_matrix_new[:] = temperature
        self.temp_matrix_old[:] = temperature


    def set_boundaries(self, left, right, top, bottom, boundary_type):
        """
        Sets the boundary conditions of the grid
        :param left: Value on left boundary
        :param right: Value on right boundary
        :param top: Value on top boundary
        :param bottom: Value on bottom boundary
        :param boundary_type: The boundary type to set
        :return: None
        """
        if boundary_type == "dirichlet":
            self.temp_matrix_old[:, 0] = left
            self.temp_matrix_old[:, -1] = right
            self.temp_matrix_old[0, :] = bottom
            self.temp_matrix_old[-1, :] = top
            self.temp_matrix_new[:, 0] = left
            self.temp_matrix_new[:, -1] = right
            self.temp_matrix_new[0, :] = bottom
            self.temp_matrix_new[-1, :] = top

        elif boundary_type == "zero_gradient_neumann":
            self.temp_matrix_old[0, :] = self.temp_matrix_old[1, :]
            self.temp_matrix_old[-1, :] = self.temp_matrix_old[-2, :]
            self.temp_matrix_old[:, 0] = self.temp_matrix_old[:, 1]
            self.temp_matrix_old[:, -1] = self.temp_matrix_old[:, -2]
            self.temp_matrix_new[0, :] = self.temp_matrix_new[1, :]
            self.temp_matrix_new[-1, :] = self.temp_matrix_new[-2, :]
            self.temp_matrix_new[:, 0] = self.temp_matrix_new[:, 1]
            self.temp_matrix_new[:, -1] = self.temp_matrix_new[:, -2]

    def create_alpha_matrix(self, kx, ky, rho, c):
        """
        Create the alpha matrix based on the parameters inputted
        :param kx: Diffusitivity in x direction
        :param ky: Diffusitivity in y direction
        :param rho: Density
        :param c: Specific heat capacity
        :return: None
        """
        self.rho_matrix = np.ones_like(self.temp_matrix_old) * rho
        self.c_matrix = np.ones_like(self.temp_matrix_old) * c

        self.kx_matrix = np.ones_like(self.temp_matrix_old) * kx
        self.ky_matrix = np.ones_like(self.temp_matrix_old) * ky

        self.alpha_x_matrix = self.kx_matrix / (self.rho_matrix * self.c_matrix)
        self.alpha_y_matrix = self.ky_matrix / (self.rho_matrix * self.c_matrix)

    def set_material_region(self, region_mask_fn, kx=None, ky=None, rho=None, c=None):
        """
        Set the alpha matrix of a specific region based on a function given by the user
        :param kx: Diffusitivity in x direction
        :param ky: Diffusitivity in y direction
        :param rho: Density
        :param c: Specific heat capacity
        :param region_mask_fn: The region of the grid the user will set a new alpha value
        :return: None
        """
        mask = region_mask_fn(self.X, self.Y)
        if kx is not None:
            self.alpha_x_matrix[mask] = kx
        if ky is not None:
            self.alpha_y_matrix[mask] = ky
        if rho is not None:
            self.rho_matrix[mask] = rho
        if c is not None:
            self.c_matrix[mask] = c

        self.alpha_x_matrix[mask] = self.kx_matrix[mask] / (self.rho_matrix[mask] * self.c_matrix[mask])
        self.alpha_y_matrix[mask] = self.ky_matrix[mask] / (self.rho_matrix[mask] * self.c_matrix[mask])

    def add_source(self, source):
        """
        Add a new thermal source to the grid
        :param source: The thermal source to be added
        :return: None
        """
        self.sources.append(source)

    def update(self, t):
        """
        Updates the temperature matrix using the explicit finite difference method and thermal source values
        :param t: Current time in the simulation
        :return: None
        """
        T = self.temp_matrix_old
        T_new = self.temp_matrix_new

        lap_x = (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]) / self.dx ** 2
        lap_y = (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / self.dy ** 2

        alpha_x = self.alpha_x_matrix[1:-1, 1:-1]
        alpha_y = self.alpha_y_matrix[1:-1, 1:-1]

        T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + self.dt * (alpha_x * lap_x + alpha_y * lap_y)

        source_sum = np.zeros_like(self.temp_matrix_old)

        """
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(source.get_value, self.X, self.Y, t) for source in self.sources]

            for future in as_completed(futures):
                source_sum += future.result()
        """

        for source in self.sources:
            source_sum += source.get_value(self.X, self.Y, t)

        T_new += self.dt * source_sum
        self.temp_matrix_old, self.temp_matrix_new = self.temp_matrix_new, self.temp_matrix_old


if __name__ == "__main__":

    g = Grid(
        width=1,
        height=1,
        step=0.01,
        dt=1e-3,
        nt=500
    )

    # Set initial and boundary temperature
    g.set_initial_temperature(temperature=0)

    g.set_boundaries(
        left=0,
        right=0,
        top=0,
        bottom=0,
        boundary_type="dirichlet"
    )

    # Use water properties
    g.create_alpha_matrix(kx=1, ky=2, rho=1000, c=4180)


    source_power = 1000  # W/m² (1000 similar to sunlight intensity)
    source_area = np.pi * (0.02)**2  # sigma = 0.02 = 2cm diameter, change area formula depending on source
    total_energy_per_second = source_power * source_area  # in watts (J/s)

    # Compute A so that dT = Q / (m * c), and m = ρ * V
    mass = 1000 * source_area * 0.01  # set depth of grid = 1 cm (only used to model source) , 1000 = density of water
    dT_per_second = total_energy_per_second / (mass * 4180)  # °C/s, 4180 = specific heat capacity of water

    g.add_source(ThermalSource(
        source_type="stationary_gaussian",
        A=dT_per_second,
        center=(0.5, 0.5),
        sigma=0.02
    ))

    g.add_source(ThermalSource(
        source_type="uniform",
        A=-0.001,
        center=(0.5, 0.5),
    ))


    def animate_func_fixed_scale(frame):
        current_time = frame * g.dt  # convert frame count to seconds
        g.update(t=current_time)
        im.set_array(g.temp_matrix_old)
        return [im]


    fig, ax = plt.subplots()
    im = ax.imshow(
        g.temp_matrix_old,
        cmap='hot',
        interpolation='nearest',
        vmin=0,
        vmax=100
    )
    plt.colorbar(im, ax=ax)

    ani = animation.FuncAnimation(fig, animate_func_fixed_scale, frames=g.nt, interval=50, blit=True)
    plt.show()
