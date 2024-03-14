import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import pandas as pd
import ast
from scipy.stats import qmc
from pyDOE2 import fullfact
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# plt.style.use('seaborn-v0_8-bright')
from ipywidgets import widgets
import ast
from vr_app_functions import plot_target_contour, DownstreamProcess, response_surface
from IPython.display import Markdown
from importlib import reload
import vr_app_functions
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def response_surface(x, y, par, scale=None, target_range=None):
    def response_model(x, y, a, b, c, d, e, f):
        return a + b*x + c*y + d*x*y + e*x**2 + f*y**2
    
    design_space = {
        'x': [x.min(), x.max()],
        'y': [y.min(), y.max()],
        'coef': par,
        'scale_xy': scale,
        'target_range': target_range
    }

    if scale is None:
        pass
    if scale == 'minmax':
        xmin, ymin = x.min(), y.min()
        xmax, ymax = x.max(), y.max()
        x = (x - xmin) / (xmax - xmin)
        y = (y - ymin) / (ymax - ymin)
    if scale == 'absmax':
        xmin, ymin = x.min(), y.min()
        xmax, ymax = x.max(), y.max()
        x = 2 * (x - xmin) / (xmax - xmin) - 1
        y = 2 * (y - ymin) / ( ymax - ymin) - 1

    x, y = np.meshgrid(x, y)
    z = response_model(x, y, *par)
    zmin, zmax = z.min(), z.max()

    if target_range is not None:    
        z = (target_range[1] - target_range[0]) * (z - zmin) / (zmax - zmin) + target_range[0]

    # rescale to original scale
    if scale is None:
        pass
    if scale == 'minmax':
        x = x * (xmax - xmin) + xmin
        y = y * (ymax - ymin) + ymin
    if scale == 'absmax':
        x = (x + 1) * (xmax - xmin) / 2 + xmin
        y = (y + 1) * (ymax - ymin) / 2 + ymin
    
    design_space['z'] = (zmin, zmax)

    return x, y, z, design_space


def plot_target_contour(x, y, z, xlabel, ylabel, zlabel, ax):

    # Plot the surface.
    cp = ax.contourf(x, y, z, cmap=cm.viridis)
    plt.colorbar(cp, label=zlabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)




def parameter_sampling(data, n_samples=1000, seed=12345, sampling_type='LHS', doe_levels=5):

    def get_parameter_ranges(data):
        ranges = {}
        processes = data['Step'].unique()

        for p in processes:
            ranges[p] = {}
            x_name = data.loc[data['Step'] == p].iloc[0]['x_name']
            ranges[p][x_name] = ast.literal_eval(data.loc[data['Step'] == p].iloc[0]['x'])

            y_name = data.loc[data['Step'] == p].iloc[0]['y_name']
            ranges[p][y_name] = ast.literal_eval(data.loc[data['Step'] == p].iloc[0]['y'])

        return ranges
    
    def get_parameter_list(ranges):
        parameter_list = [] 
        for key in ranges.keys():
            for key2 in ranges[key].keys():
                parameter_list.append(key2)

        return parameter_list
    
    def get_boundaries(ranges):
        lb, ub = [], []
        for key in ranges.keys():
            for key2 in ranges[key].keys():
                ub.append(ranges[key][key2][1])
                lb.append(ranges[key][key2][0])

        return lb, ub
    
    def scale_ranges_doe(samples, ranges):
        n_par = samples.shape[1]

        i = 0
        for process in ranges.keys():
            for parameter in ranges[process].keys():
                range = ranges[process][parameter]
                dist = range[1] - range[0]

                samples[:, i] = samples[:, i]*dist + range[0]
                i += 1
        return samples
    
    ranges = get_parameter_ranges(data)
    par_list = get_parameter_list(ranges)
    n_par = len(par_list)
    lb, ub = get_boundaries(ranges)


    if sampling_type == 'LHS':
        sampler = qmc.LatinHypercube(d=n_par,seed=seed)
        samples = sampler.random(n=int(n_samples))
        samples = qmc.scale(samples, lb, ub)

    if sampling_type == 'DOE':
        if isinstance(doe_levels, int):
            levels = [doe_levels for i in range(n_par)]
        if isinstance(doe_levels, float):
            levels = [int(doe_levels) for i in range(n_par)]
        if isinstance(doe_levels, list):
            levels = doe_levels

        samples = fullfact(levels)
        samples = MinMaxScaler().fit_transform(samples)
        samples = scale_ranges_doe(samples, ranges)
        
    samples_dict = {}
    i = 0
    for process in ranges.keys():
        samples_dict[process] = {}
        for parameter in ranges[process].keys():
            samples_dict[process][parameter] = samples[:, i]
            i += 1
        
    return samples_dict, samples.shape[0]



class DownstreamProcess:
    def __init__(self, initial_conditions):
        self.initial_process_conditions = initial_conditions
        self.current_process_conditions = {}

        for key in initial_conditions.keys():
            self.current_process_conditions[key] = self.initial_process_conditions[key]
                 
    def response_model(self, x, y, a, b, c, d, e, f):
        return a + b*x + c*y + d*x*y + e*x**2 + f*y**2

    def simulate_single_unit_operation(self, x, y, design_space, target):
        design_space = design_space.loc[target]
        
        if design_space['scale_xy'] == 'minmax':
            x_range = ast.literal_eval(design_space['x'])
            y_range = ast.literal_eval(design_space['y'])
            xmin, xmax = x_range
            ymin, ymax = y_range
            x = (x - xmin) / (xmax - xmin)
            y = (y - ymin) / (ymax - ymin)
       
        if design_space['scale_xy'] == 'absmax':
            x_range = ast.literal_eval(design_space['x'])
            y_range = ast.literal_eval(design_space['y'])
            xmin, xmax = x_range
            ymin, ymax = y_range
            x = 2 * (x - xmin) / (xmax - xmin) - 1
            y = 2 * (y - ymin) / ( ymax - ymin) - 1 
        
        
        z = self.response_model(x, y, *ast.literal_eval(design_space['coef']))
        
        target_range = design_space['target_range']
        if isinstance(target_range, type(np.nan)):
            target_range = None
        else:
            target_range = ast.literal_eval(target_range)
        if target_range is not None:
           tr = target_range
           zmin, zmax = ast.literal_eval(design_space['z'])
           z = (tr[1] - tr[0]) * (z - zmin) / (zmax - zmin) + tr[0] 

        
        self.current_process_conditions[target] = self.current_process_conditions[target]*z


    def update_product_concentration(self):

        self.current_process_conditions['Product concentration'] = self.initial_process_conditions['Product concentration'] * self.initial_process_conditions['Volume'] * self.current_process_conditions['Yield'] / self.current_process_conditions['Volume']
