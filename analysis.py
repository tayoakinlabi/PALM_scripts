# Script used to produce the figures in Akinlabi et al. 2026 - Linking canyon air temperature response to energy forcing: A large-eddy simulation study of cool roof effects

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib import cm
import math
from netCDF4 import Dataset
#import seaborn as sns
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.colors as colors
from matplotlib.legend_handler import HandlerTuple

# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def average_y(data):
    return np.mean(data,axis = 1)

def tke_func(u, v, w, uu, vv, ww):
    uur = calc_fluxes(uu, u, u)
    vvr = calc_fluxes(vv, v, v)
    wwr = calc_fluxes(ww, w, w)
    return 0.5*(uur + vvr + wwr)

def get_range(i):
    if i == 0:
        return {'m': -2, 'n': 4.01, 'xs': 2, 'ticks': [-2, 0, 2, 4], 'label': r"$\langle\overline{u}\rangle_y$ [m s$^{-1}$]"}
    elif i == 1:
        return {'m': -2, 'n': 2.01, 'xs': 4, 'ticks': [-2, -1, 0, 1, 2], 'label': r"$\langle\overline{w}\rangle_y$ [m s$^{-1}$]"}
    elif i == 2:
        return {'m': 0, 'n': 12.01, 'xs': 8, 'ticks': [0, 2, 4, 6, 8., 10., 12.], 'label': r"$\langle\overline{\theta}\rangle_y - \theta_0$ [K]"}
    else:
        return {'m': 0, 'n': 0.801, 'xs': 16, 'ticks': [0, 0.2, 0.4, 0.6, 0.8], 'label': r"$\langle {TKE} \rangle_y$ [m$^{2}$ s$^{-2}$]"}


# plot in figure 2
def variable_plot_uw(folders, coords, let_u, let_w, is_mesh=True):
    """
    Combined plot of 'u' (rows 0-1) and 'w' (rows 2-3).
    Each variable spans 2 rows x 2 cols (4 folders total).
    Separate colorbars on the right for u and w.
    """
    B = 10
    xs_center = [4, 8, 16, 32]

    fig, axs = plt.subplots(4, 2, sharey=False, figsize=(14., 11.))
    plt.rcParams["font.size"] = "12"

    cf_u = None
    cf_w = None

    for axy in range(len(folders)):
        coord = coords[axy]
        ind = 4 if axy == 1 else 1

        nc_path = f"{folders[axy]}/OUTPUT/{folders[axy]}_av_3d.00{ind}.nc"

        with Dataset(nc_path, mode='r') as ds:
            x     = ds.variables['x'][coord[0]:coord[1]]
            z     = ds.variables['zw_3d'][:coord[3]]
            u_raw = ds.variables['u'][-1, :coord[3], :, coord[0]:coord[1]]
            w_raw = ds.variables['w'][-1, :coord[3], :, coord[0]:coord[1]]

        u_var = average_y(u_raw)
        w_var = average_y(w_raw)

        x_norm = (x - x[16 + xs_center[axy]]) / B
        z_norm = z / B

        rng_u = get_range(0)
        rng_w = get_range(1)

        # Map folder index to (row, col) within each variable's 2x2 block
        # axy 0,1 -> row 0, cols 0,1  |  axy 2,3 -> row 1, cols 0,1
        row_in_block = axy // 2
        col           = axy % 2

        ax_u = row_in_block          # u occupies rows 0-1
        ax_w = row_in_block + 2      # w occupies rows 2-3

        if is_mesh:
            U_norm = u_var
            W_norm = w_var

        # ================================================================
        # U rows (0-1)
        # ================================================================
        cf = axs[ax_u, col].contourf(
            x_norm, z_norm, u_var,
            levels=np.arange(rng_u['m'], rng_u['n'], 0.01),
            cmap='bwr',
            norm=MidpointNormalize(midpoint=0., vmin=rng_u['m'], vmax=rng_u['n'])
        )
        cf_u = cf

        if is_mesh:
            axs[ax_u, col].streamplot(
                x_norm, z_norm, U_norm, W_norm,
                density=(4, 1.5), linewidth=0.3, arrowsize=0.4, color='k'
            )

        axs[ax_u, col].set_facecolor("grey")
        axs[ax_u, col].set_xticks([])
        if col == 0:
            axs[ax_u, col].set_ylabel('z/H')
        dh = 0.1 + x_norm[0]
        axs[ax_u, col].text(dh, 1.53, let_u[axy], fontsize=12)

        # ================================================================
        # W rows (2-3)
        # ================================================================
        cf = axs[ax_w, col].contourf(
            x_norm, z_norm, w_var,
            levels=np.arange(rng_w['m'], rng_w['n'], 0.01),
            cmap='bwr',
            norm=MidpointNormalize(midpoint=0., vmin=rng_w['m'], vmax=rng_w['n'])
        )
        cf_w = cf

        if is_mesh:
            axs[ax_w, col].streamplot(
                x_norm, z_norm, U_norm, W_norm,
                density=(4, 1.5), linewidth=0.3, arrowsize=0.4, color='k'
            )

        axs[ax_w, col].set_facecolor("grey")
        axs[ax_w, col].set_xticks([])
        if col == 0:
            axs[ax_w, col].set_ylabel('z/H')
        axs[ax_w, col].text(dh, 1.53, let_w[axy], fontsize=12)

    # ---- Separate colorbars on the right, vertically aligned to their rows ----
    cb_u = fig.colorbar(cf_u, ax=axs[:2, :].ravel().tolist(), location='right',
                        shrink=0.6, pad=0.01)
    cb_u.set_ticks(rng_u['ticks'])
    cb_u.set_label(rng_u['label'], fontsize=14)

    cb_w = fig.colorbar(cf_w, ax=axs[2:, :].ravel().tolist(), location='right',
                        shrink=0.6, pad=0.01)
    cb_w.set_ticks(rng_w['ticks'])
    cb_w.set_label(rng_w['label'], fontsize=14)

    plt.savefig(os.path.join(save_path, 'u_w.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
folders = ['c2_urban_canyon', 'c4_urban_canyon', 'c5_urban_canyon', 'c6_urban_canyon']
coords  = [[28,68,288,25], [40,88,288,25], [64,128,288,25], [8,104,288,25]]

let_u = ['a) AR = 2', 'b) AR = 1', 'c) AR = 0.5', 'd) AR = 0.25']
let_w = ['e) AR = 2', 'f) AR = 1', 'g) AR = 0.5', 'h) AR = 0.25']

# Run the function
#variable_plot_uw(folders, coords, let_u, let_w, is_mesh=True)


def variable_plot_theta_tke(folders, coords, let_theta, let_tke, is_mesh=False):
    """
    Combined plot of 'theta' (rows 0-1) and 'tke' (rows 2-3).
    Each variable spans 2 rows x 2 cols (4 folders total).
    Separate colorbars on the right for theta and tke.
    """
    B = 10
    xs_center = [4, 8, 16, 32]

    fig, axs = plt.subplots(4, 2, sharey=True, figsize=(14., 11.))
    plt.rcParams["font.size"] = "12"

    cf_theta = None
    cf_tke   = None

    for axy in range(len(folders)):
        coord = coords[axy]
        ind = 4 if axy == 1 else 1

        nc_path = f"{folders[axy]}/OUTPUT/{folders[axy]}_av_3d.00{ind}.nc"

        with Dataset(nc_path, mode='r') as ds:
            x       = ds.variables['x'][coord[0]:coord[1]]
            z       = ds.variables['zw_3d'][:coord[3]]

            # theta
            theta_raw = ds.variables['theta'][-1, :coord[3], :, coord[0]:coord[1]]

            # tke components
            u_raw  = ds.variables['u'][-1, :coord[3], :, coord[0]:coord[1]]
            v_raw  = ds.variables['v'][-1, :coord[3], :, coord[0]:coord[1]]
            w_raw  = ds.variables['w'][-1, :coord[3], :, coord[0]:coord[1]]
            uu_raw = ds.variables['uu_product'][-1, :coord[3], :, coord[0]:coord[1]]
            vv_raw = ds.variables['vv_product'][-1, :coord[3], :, coord[0]:coord[1]]
            ww_raw = ds.variables['ww_product'][-1, :coord[3], :, coord[0]:coord[1]]

        # Apply the same uu fix as in original
        if axy == 0:
            for k in range(17):
                uu_raw[k, :, 16] = uu_raw[k, :, 17]

        theta_var = average_y(theta_raw - 300)
        tke_var   = tke_func(u_raw, v_raw, w_raw, uu_raw, vv_raw, ww_raw)
        tke_var   = average_y(tke_var)

        x_norm = (x - x[16 + xs_center[axy]]) / B
        z_norm = z / B

        rng_theta = get_range(2)
        rng_tke   = get_range(3)

        row_in_block = axy // 2
        col          = axy % 2

        ax_theta = row_in_block        # theta occupies rows 0-1
        ax_tke   = row_in_block + 2    # tke occupies rows 2-3

        # ================================================================
        # Theta rows (0-1)
        # ================================================================
        cf = axs[ax_theta, col].contourf(
            x_norm, z_norm, theta_var,
            levels=np.arange(rng_theta['m'], rng_theta['n'], 0.01),
            cmap='bwr',
            norm=MidpointNormalize(midpoint=0., vmin=rng_theta['m'], vmax=rng_theta['n'])
        )
        cf_theta = cf

        if is_mesh:
            U_norm = average_y(u_raw)
            W_norm = average_y(w_raw)
            axs[ax_theta, col].streamplot(
                x_norm, z_norm, U_norm, W_norm,
                density=(4, 1.5), linewidth=0.3, arrowsize=0.4, color='k'
            )

        axs[ax_theta, col].set_facecolor("grey")
        axs[ax_theta, col].set_xticks([])
        if col == 0:
            axs[ax_theta, col].set_ylabel('z/H')
        dh = 0.1 + x_norm[0]
        axs[ax_theta, col].text(dh, 1.53, let_theta[axy], fontsize=12)

        # ================================================================
        # TKE rows (2-3)
        # ================================================================
        cf = axs[ax_tke, col].contourf(
            x_norm, z_norm, tke_var,
            levels=np.arange(rng_tke['m'], rng_tke['n'], 0.01),
            cmap='bwr',
            norm=MidpointNormalize(midpoint=0., vmin=rng_tke['m'], vmax=rng_tke['n'])
        )
        cf_tke = cf

        if is_mesh:
            U_norm = average_y(u_raw)
            W_norm = average_y(w_raw)
            axs[ax_tke, col].streamplot(
                x_norm, z_norm, U_norm, W_norm,
                density=(4, 1.5), linewidth=0.3, arrowsize=0.4, color='k'
            )

        axs[ax_tke, col].set_facecolor("grey")
        axs[ax_tke, col].set_xticks([])
        if col == 0:
            axs[ax_tke, col].set_ylabel('z/H')
        axs[ax_tke, col].text(dh, 1.53, let_tke[axy], fontsize=12)

    # ---- Separate colorbars on the right, vertically aligned to their rows ----
    cb_theta = fig.colorbar(cf_theta, ax=axs[:2, :].ravel().tolist(), location='right',
                            shrink=0.6, pad=0.01)
    cb_theta.set_ticks(rng_theta['ticks'])
    cb_theta.set_label(rng_theta['label'], fontsize=14)

    cb_tke = fig.colorbar(cf_tke, ax=axs[2:, :].ravel().tolist(), location='right',
                          shrink=0.6, pad=0.01)
    cb_tke.set_ticks(rng_tke['ticks'])
    cb_tke.set_label(rng_tke['label'], fontsize=14)

    plt.savefig(os.path.join(save_path, 'theta_tke.png'), dpi=300, bbox_inches='tight')
    plt.show()

folders = ['c2_urban_canyon', 'c4_urban_canyon', 'c5_urban_canyon', 'c6_urban_canyon']
coords  = [[28,68,288,25], [40,88,288,25], [64,128,288,25], [8,104,288,25]]

let_theta = ['a) AR = 2', 'b) AR = 1', 'c) AR = 0.5', 'd) AR = 0.25']
let_tke   = ['e) AR = 2', 'f) AR = 1', 'g) AR = 0.5', 'h) AR = 0.25']

#run the function for produce figure 3
#variable_plot_theta_tke(folders, coords, let_theta, let_tke, is_mesh=False)



# function to produce figure 4

def get_range(i):
    if i == 0:
        return {'m': -0.1, 'n': 0.201, 'xs': 2, 'ticks': [-0.1, 0, 0.1, 0.2], 'label': "$\Delta \overline{u}$ [m s$^{-1}$]"}
    elif i == 1:
        return {'m': -0.04, 'n': 0.041, 'xs': 4, 'ticks': [-0.04, -0.02, 0, 0.02, 0.04], 'label': "$\Delta \overline{w}$ [m s$^{-1}$]"}
    elif i == 2:
        return {'m': -0.8, 'n': 0.01, 'xs': 8, 'ticks': [-0.8, -0.6, -0.4, -0.2, 0.], 'label': "$\Delta \overline{\\theta}$ [K]"}
    else:
        return {'m': -0.02, 'n': 0.041, 'xs': 16, 'ticks': [-0.02, 0, 0.02, 0.04], 'label': "$\Delta$ TKE [m$^{2}$ s$^{-2}$]"}
    

B = 10
fig, axs = plt.subplots(2, 2,figsize=(12., 4.))
fig.subplots_adjust(wspace=0.2, hspace=0.2)
folders = ['c4_urban_canyon']
#folders = ['4_c05_urban_canyon','4_c1_urban_canyon','4_c2_urban_canyon','4_c4_urban_canyon','4_c5_urban_canyon','4_c6_urban_canyon']
coord = [40,89,288,25]
ind = 4
ax = 0
x = Dataset(folders[ax]+'/OUTPUT/'+folders[ax]+'_av_3d.00'+str(ind)+'.nc', mode='r').variables['x'][coord[0]:coord[1]]
z = Dataset(folders[ax]+'/OUTPUT/'+folders[ax]+'_av_3d.00'+str(ind)+'.nc', mode='r').variables['zw_3d'][:coord[3]]
u = Dataset(folders[ax]+'/OUTPUT/'+folders[ax]+'_av_3d.00'+str(ind)+'.nc', mode='r').variables['u'][-1,:coord[3],:,coord[0]:coord[1]]
v = Dataset(folders[ax]+'/OUTPUT/'+folders[ax]+'_av_3d.00'+str(ind)+'.nc', mode='r').variables['v'][-1,:coord[3],:,coord[0]:coord[1]]
w = Dataset(folders[ax]+'/OUTPUT/'+folders[ax]+'_av_3d.00'+str(ind)+'.nc', mode='r').variables['w'][-1,:coord[3],:,coord[0]:coord[1]]
theta = Dataset(folders[ax]+'/OUTPUT/'+folders[ax]+'_av_3d.00'+str(ind)+'.nc', mode='r').variables['theta'][-1,:coord[3],:,coord[0]:coord[1]]
uu = Dataset(folders[ax]+'/OUTPUT/'+folders[ax]+'_av_3d.00'+str(ind)+'.nc', mode='r').variables['uu_product'][-1,:coord[3],:,coord[0]:coord[1]]
vv = Dataset(folders[ax]+'/OUTPUT/'+folders[ax]+'_av_3d.00'+str(ind)+'.nc', mode='r').variables['vv_product'][-1,:coord[3],:,coord[0]:coord[1]]
ww = Dataset(folders[ax]+'/OUTPUT/'+folders[ax]+'_av_3d.00'+str(ind)+'.nc', mode='r').variables['ww_product'][-1,:coord[3],:,coord[0]:coord[1]]

ind = 3
u1 = Dataset('4_'+folders[ax]+'/OUTPUT/'+'4_'+folders[ax]+'_av_3d.00'+str(ind)+'.nc', mode='r').variables['u'][-1,:coord[3],:,coord[0]:coord[1]]
v1 = Dataset('4_'+folders[ax]+'/OUTPUT/'+'4_'+folders[ax]+'_av_3d.00'+str(ind)+'.nc', mode='r').variables['v'][-1,:coord[3],:,coord[0]:coord[1]]
w1 = Dataset('4_'+folders[ax]+'/OUTPUT/'+'4_'+folders[ax]+'_av_3d.00'+str(ind)+'.nc', mode='r').variables['w'][-1,:coord[3],:,coord[0]:coord[1]]
theta1 = Dataset('4_'+folders[ax]+'/OUTPUT/'+'4_'+folders[ax]+'_av_3d.00'+str(ind)+'.nc', mode='r').variables['theta'][-1,:coord[3],:,coord[0]:coord[1]]
uu1 = Dataset('4_'+folders[ax]+'/OUTPUT/'+'4_'+folders[ax]+'_av_3d.00'+str(ind)+'.nc', mode='r').variables['uu_product'][-1,:coord[3],:,coord[0]:coord[1]]
vv1 = Dataset('4_'+folders[ax]+'/OUTPUT/'+'4_'+folders[ax]+'_av_3d.00'+str(ind)+'.nc', mode='r').variables['vv_product'][-1,:coord[3],:,coord[0]:coord[1]]
ww1 = Dataset('4_'+folders[ax]+'/OUTPUT/'+'4_'+folders[ax]+'_av_3d.00'+str(ind)+'.nc', mode='r').variables['ww_product'][-1,:coord[3],:,coord[0]:coord[1]]
        
tke = tke_func(u, v, w, uu, vv, ww)
tke1 = tke_func(u1, v1, w1, uu1, vv1, ww1)
var_list = [u1 - u, w1-w, theta1-theta, tke1-tke]

var = average_y(var_list[0])
m = get_range(0)['m']#np.nanmin(var) #
n = get_range(0)['n']#np.nanmax(var) #
mid_val = 0.
bh_contour = axs[0,0].contourf( (x-x[16+get_range(2)['xs']])/B, z/B, var,levels=np.arange(m, n, 0.001),cmap='bwr', norm=MidpointNormalize(midpoint=mid_val,vmin=m, vmax=n))
axs[0,0].set_facecolor("grey")
axs[0,0].text(-0., 1.53, 'a)', fontsize = 14)
axs[0,0].set_xlim([-1.5, 1.5])
axs[0,0].set_yticks([0, 0.5, 1., 1.5])
axs[0,0].set_xticks([])
bh_colorbar = fig.colorbar(bh_contour,location='right',shrink=0.95, pad=0.06)
bh_colorbar.set_ticks(get_range(0)['ticks'])
bh_colorbar.set_label(get_range(0)['label'], fontsize = 14)

var = average_y(var_list[1])
m = get_range(1)['m']#np.nanmin(var) #
n = get_range(1)['n']#np.nanmax(var) #
bh_contour = axs[0,1].contourf( (x-x[16+get_range(2)['xs']])/B, z/B, var,levels=np.arange(m, n, 0.001),cmap='bwr', norm=MidpointNormalize(midpoint=mid_val,vmin=m, vmax=n))
axs[0,1].set_facecolor("grey")
axs[0,1].set_xlim([-1.5, 1.5])
axs[0,1].text(-0., 1.53, 'b)', fontsize = 14)
axs[0,1].set_yticks([0, 0.5, 1., 1.5])
axs[0,1].set_xticks([])
bh_colorbar = fig.colorbar(bh_contour,location='right',shrink=0.95, pad=0.06)
bh_colorbar.set_ticks(get_range(1)['ticks'])
bh_colorbar.set_label(get_range(1)['label'], fontsize = 14)

var = average_y(var_list[2])
m = get_range(2)['m']#np.nanmin(var) #
n = get_range(2)['n']#np.nanmax(var) #
bh_contour = axs[1,0].contourf( (x-x[16+get_range(2)['xs']])/B, z/B, var,levels=np.arange(m, n, 0.001),cmap='bwr', norm=MidpointNormalize(midpoint=mid_val,vmin=m, vmax=n))
axs[1,0].set_facecolor("grey")
axs[1,0].set_xlim([-1.5, 1.5])
axs[1,0].text(-0., 1.53, 'c)', fontsize = 14)
axs[1,0].set_yticks([0, 0.5, 1., 1.5])
axs[1,0].set_xticks([])
bh_colorbar = fig.colorbar(bh_contour,location='right',shrink=0.95, pad=0.06)
bh_colorbar.set_ticks(get_range(2)['ticks'])
bh_colorbar.set_label(get_range(2)['label'], fontsize = 14)

var = average_y(var_list[3])
m = get_range(3)['m']#np.nanmin(var) #
n = get_range(3)['n']#np.nanmax(var) #
bh_contour = axs[1,1].contourf( (x-x[16+get_range(2)['xs']])/B, z/B, var,levels=np.arange(m, n, 0.001),cmap='bwr', norm=MidpointNormalize(midpoint=mid_val,vmin=m, vmax=n))
axs[1,1].set_facecolor("grey")
axs[1,1].set_xlim([-1.5, 1.5])
axs[1,1].text(-0., 1.53, 'd)', fontsize = 14)
axs[1,1].set_yticks([0, 0.5, 1., 1.5])
axs[1,1].set_xticks([])
bh_colorbar = fig.colorbar(bh_contour,location='right',shrink=0.95, pad=0.06)
bh_colorbar.set_ticks(get_range(3)['ticks'])
bh_colorbar.set_label(get_range(3)['label'], fontsize = 14)
plt.savefig(os.path.join(save_path,f'delta_combined.png'), dpi=300, bbox_inches='tight')


#Use this code for figure 5 & 6

from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
from netCDF4 import Dataset

def get_flux(text):
    flux = text.split('=')[1].split(' ')[1].split('$')[0]
    return float(flux)

def variable_plot(folders, coords, variable, var_range, let, is_mesh=True, save_path="."):
    B = 10
    labels = ['$Q_{r} = 0.1125$ K m/s', '$Q_{r} = 0.075$ K m/s',
              '$Q_{r} = 0.0375$ K m/s', '$Q_{r} = 0.01875$ K m/s']

#     let_bottom = ['g)', 'h)', 'i)', 'j)', 'k)', 'l)']
    let_bottom = ['e)', 'f)', 'g)', 'h)']
    plt.rcParams["font.size"] = "10"

    # Helper function for plotting each subplot
    def plot_subplot(ax, axy, coord, base_path, theta_base, labels, z, z_scale, rf=None, second_half=False):
        for i in range(4):
            ind = 1 if not (axy == 5 and i == 3) else 4
            case_path = f"{i+1}_{base_path}"
            theta = Dataset(f"{case_path}/OUTPUT/{case_path}_av_3d.00{ind}.nc",
                            mode='r').variables['theta'][-1, :coord[3], :, :]

            if second_half:
                base_flux = get_flux("$Q_{r} = 0.15$ K m/s")
                case_flux = get_flux(labels[i])
                scaled_profile = profile(theta - theta_base) * (1 - rf) / (1.225*1005*(case_flux - base_flux) * rf)
                ax.plot(scaled_profile, z / z_scale, label=labels[i])
            else:
                ax.plot(profile(theta - theta_base), z / z_scale, label=labels[i])

        # y-label only for first column
        if axy % 2 == 0:
            ax.set_ylabel('z/H')
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

        if axy >= 2:
            if second_half:
                ax.set_xlabel(r"$\frac{\langle\Delta \overline{\theta}\rangle_{xy}}{\rho c_p \Delta Q_{r} \frac{RF}{1 - RF}}$ [K W$^{-1}$m$^2$]", fontsize=12)
            else: 
                ax.set_xlabel(r"$\langle\Delta \overline{\theta}\rangle_{xy}$ [K]", fontsize=12)
        ax.grid()

        if second_half and axy == 3:
            ax.legend(fontsize=10, loc='best')
        if not second_half and axy == 3:
            ax.legend(fontsize=10, loc='best')

    # -------------------------
    # Figure 1: raw Δθ
    # -------------------------
    fig1 = plt.figure(figsize=(8, 6))
    gs_top = gridspec.GridSpec(2, 2, figure=fig1, hspace=0.35)  # <-- add space between rows
    axs_top = [fig1.add_subplot(gs_top[idx // 2, idx % 2]) for idx in range(4)]
    len_range = [ -0.6, -0.5, -0.3, -0.2]
    for i, ax in enumerate(axs_top):
        coord = coords[i]
        ind, ind1 = (1, 4) if i == 1 else (1, 1)
        base_path = folders[i]
        z = Dataset(f"{base_path}/OUTPUT/{base_path}_av_3d.00{ind}.nc",
                    mode='r').variables['zw_3d'][:coord[3]]
        theta_base = Dataset(f"{base_path}/OUTPUT/{base_path}_av_3d.00{ind1}.nc",
                             mode='r').variables['theta'][-1, :coord[3], :, :]
        plot_subplot(ax, i, coord, base_path, theta_base, labels, z, B, second_half=False)
        ax.set_xlim([len_range[i], 0])
        ax.set_yticks([0, 0.5,1,1.5])
        ax.set_title(let[i], fontsize=10, pad=8)

    fig1.savefig(os.path.join(save_path, f"{variable}_raw.png"), dpi=300, bbox_inches='tight')

    # -------------------------
    # Figure 2: scaled Δθ
    # -------------------------
    fig2 = plt.figure(figsize=(8, 6))
    gs_bottom = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.35)  # <-- add space between rows
    axs_bottom = [fig2.add_subplot(gs_bottom[idx // 2, idx % 2]) for idx in range(4)]
    xs_center = [2./3., 1./2., 1./3., 150./720.]
    len_range = [ 2., 3, 4, 6]
    for i, ax in enumerate(axs_bottom):
        coord = coords[i]
        rf = xs_center[i]
        ind, ind1 = (1, 4) if i == 1 else (1, 1)
        base_path = folders[i]
        z = Dataset(f"{base_path}/OUTPUT/{base_path}_av_3d.00{ind}.nc",
                    mode='r').variables['zw_3d'][:coord[3]]
        theta_base = Dataset(f"{base_path}/OUTPUT/{base_path}_av_3d.00{ind1}.nc",
                             mode='r').variables['theta'][-1, :coord[3], :, :]
        plot_subplot(ax, i, coord, base_path, theta_base, labels, z, B, rf=rf, second_half=True)
        ax.set_xlim([0, len_range[i]*0.001])
        ax.set_yticks([0, 0.5,1,1.5])
        ax.set_title(let[i], fontsize=10, pad=8)
        ax.ticklabel_format(style='sci',scilimits=(-2,2),axis='x')
        ax.xaxis.major.formatter._useMathText = True

    fig2.savefig(os.path.join(save_path, f"{variable}_scaled.png"), dpi=300, bbox_inches='tight')

    
# folders = ['c05_urban_canyon','c1_urban_canyon','c2_urban_canyon','c4_urban_canyon','c5_urban_canyon','c6_urban_canyon']
folders = ['c2_urban_canyon','c4_urban_canyon','c5_urban_canyon','c6_urban_canyon']
coords = [[28,68,288,25],[40,88,288,25],[64,128,288,25],[8,104,288,25]]
var_range = 0
variable = 'diff_theta_profile'
let = ['a) AR = 2','b) AR = 1','c) AR = 0.5','d) AR = 0.25']
variable_plot(folders, coords, variable, var_range, let, is_mesh = False)

