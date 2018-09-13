import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpp


class DerenzoPhantom(object):
    """
    Usage:
    radius = 37.0
    well_seps = (1.0, 6.0, 5.0, 4.0, 3.0, 2.0)
    my_phantom = DerenzoPhantom(radius, well_seps)
    my_phantom.show()
    """
    _num_sections = 6

    def __init__(self, radius, well_separations, cyl_height=0, unit="mm"):
        self.radius = radius
        self.well_seps = well_separations
        self.depth = cyl_height
        self.unit = unit

        # Define sections
        self.sections = []
        for well_sep, rot_angle in zip(self.well_seps,
                                       np.arange(0, 360., 360. / self._num_sections)):
            section = DerenzoSection(self.radius, well_sep)
            section.apply_rotation(rot_angle)
            self.sections.append(section)

        # Initialize graphic (values hard-coded for now)
        self.fig = plt.figure(figsize=(16, 16))
        self.fig.set_dpi(16)
        self.ax = self.fig.gca()
        self.cyl_patch = mpp.Circle((0, 0), radius=self.radius, color='gray',
                                    alpha=0.3)
        self.ax.add_patch(self.cyl_patch)
        self.ax.set_xlim((-1.2*self.radius, 1.2*self.radius))
        self.ax.set_ylim((-1.2*self.radius, 1.2*self.radius))
        self.ax.set_axis_off()
        self.ax.autoscale(enable=True, axis='both', tight=None)

        # Plot well locations from all sections of the phantom
        for section in self.sections:
            section.plot_wells(self.fig, self.ax)

    @property
    def area(self):
        return np.sum([s.total_area for s in self.sections])

    @property
    def num_wells(self):
        return np.sum([s.num_wells for s in self.sections])

    def show(self):
        self.fig.canvas.draw()
        plt.show()


class DerenzoSection(object):

    def __init__(self, phantom_radius, well_separation, section_offset=0.1):
        self.R = phantom_radius
        self.well_sep = well_separation
        self.r = self.well_sep / 2.0
        self.section_offset = self.R * section_offset
        # Determine well locations
        self.place_wells_in_section()
        # Location for section label
        self.label_xy = np.array((0, -1.1 * self.R))

    @property
    def row_height(self):
        return self.well_sep * np.sqrt(3)

    @property
    def num_rows(self):
        h_section = self.R - (2 * self.section_offset + self.well_sep)
        return int(np.floor(h_section / self.row_height))

    @property
    def num_wells(self):
        return np.sum(1 + np.arange(self.num_rows))

    @property
    def well_area(self):
        return np.pi * self.r**2

    @property
    def total_area(self):
        return self.num_wells * self.well_area

    @property
    def label(self):
        return "%.1f mm" %(self.well_sep)

    def place_wells_in_section(self):

        if self.num_rows <= 1:
            self.section_offset = 0.0
            if self.num_rows <= 1:
                warnings.warn(("Cannot fit multiple features in section with "
                               "feature size = %s" %(self.well_sep)))
        xs, ys = [], []
        for i in range(self.num_rows):
            rn = i + 1
            for x in np.arange(-rn, rn, 2) + 1:
                xs.append(x * self.well_sep)
                ys.append(-(self.section_offset + self.row_height * rn))
        self.locs = np.vstack((xs, ys)).T

    def apply_rotation(self, deg):
        """
        Rotate well locations around central (z) axis by 'deg' degrees.
        deg > 0: Counter-clockwise | deg < 0: clockwise
        """
        self.rot_angle = deg
        th = -1 * deg * (np.pi / 180)
        rot_mat = np.array([(np.cos(th), -np.sin(th)),
                            (np.sin(th),  np.cos(th))])
        # Rotate well locations
        self.locs = np.array([np.dot(l, rot_mat) for l in self.locs])
        # Rotate label location
        self.label_xy = np.dot(self.label_xy, rot_mat)

    def plot_wells(self, fig, ax):
        """
        Plot the well pattern for the given section on the input figure and
        axis handles.
        """
        # Plot wells
        for xy in self.locs:
            cyl = mpp.Circle(xy, radius=self.r)
            ax.add_patch(cyl)