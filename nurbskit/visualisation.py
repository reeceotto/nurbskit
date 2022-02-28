"""
Visualisation toolkit.

Author: Reece Otto 11/09/2020
"""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection

def path_plot_2D(path, axes=None, show_path=True, show_control_points=True, 
    show_control_net=True, show_knots=True, path_label='Path',
    control_point_label='Control Points', control_net_label='Control Net',
    knot_label='Knots', path_style='k', control_point_style='ro',
    control_net_style='b-', knot_style='gx', control_net_alpha=0.3, 
    n_points=100, **kwargs):
    """
    Returns 2D matplotlib axes with path plotted.
    
    Arguments:
        path = path object
    
    Keyword arguments:
        axes = matplotlib axes
        show_path = option to show path
        show_control_points = option to show control points
        show_control_net = option to show control net
        show_knots = option to show knots
        path_label = label of path on legend
        control_point_label = label of control points on legend
        control_net_label = label of control net on legend
        knot_label = label of knot on legend
        path_style = matplotlib style of plotted path
        control_point_style = matplotlib style of plotted control points
        control_net_style = matplotlib style of plotted control net
        knot_style = matplotlib style of plotted knots
        control_net_alpha = transparency of control net
        n_points = number of points evaluated along path
        u_i = initial u coordinate
        u_f = final u coordinate
    """

    # give warning message if curve is 3D
    if len(path.P[0]) == 3:
        print('Warning: path has 3D control points.' + \
        ' Only the x and y coordinates will be plotted.')

    # initialise axes
    if axes == None:
        axes = plt.axes()

    # plot path if desired
    if show_path == True:
        # evaluate points along curve
        u_i = kwargs.get('u_i', path.u_start())
        u_f = kwargs.get('u_f', path.u_end())
        path_coords = path.list_eval(u_i=u_i, u_f=u_f, n_points=n_points)

        # plot curve
        axes.plot(path_coords[:,0], path_coords[:,1], path_style, 
            label=path_label)
    
    # plot control points if desired
    if show_control_points == True:
        axes.plot(path.P[:,0], path.P[:,1], control_point_style, 
            label=control_point_label)
    
    # plot control net if desired
    if show_control_net == True:
        axes.plot(path.P[:,0], path.P[:,1], control_net_style, 
            alpha=control_net_alpha, label=control_net_label)
    
    # plot knots if desired
    if show_knots == True:
        knots = path.knot_eval()
        axes.plot(knots[:,0], knots[:,1], knot_style, label=knot_label)
    
    # return axes
    return axes
    
def path_plot_3D(path, axes=None, show_path=True, show_control_points=True, 
    show_control_net=True, show_knots=True, path_label='Path',
    control_point_label='Control Points', control_net_label='Control Net',
    knot_label='Knots', path_style='k', control_point_style='ro',
    control_net_style='b-', knot_style='gx', control_net_alpha=0.3, 
    n_points=100, **kwargs):
    """
    Creates 3D plot of a path.
    
    Arguments:
        path = path object
    
    Keyword arguments:
        axes = matplotlib axes (must be 3D)
        show_path = option to show path
        show_control_points = option to show control points
        show_control_net = option to show control net
        show_knots = option to show knots
        path_label = label of path on legend
        control_point_label = label of control points on legend
        control_net_label = label of control net on legend
        knot_label = label of knot on legend
        path_style = matplotlib style of plotted path
        control_point_style = matplotlib style of plotted control points
        control_net_style = matplotlib style of plotted control net
        knot_style = matplotlib style of plotted knots
        control_net_alpha = transparency of control net
        n_points = number of points evaluated along path
        u_i = initial u coordinate
        u_f = final u coordinate
    """

    # check if curve is 3D
    dims = np.nan * np.ones(len(path.P))
    for i in range(len(path.P)):
        dims[i] = len(path.P[i])
    if not np.all(dims == 3):
        raise AttributeError('Not all control points are 3D.')
    del dims

    # initialise axes
    if axes == None:
        axes = plt.axes(projection='3d')

    # plot path if desired
    if show_path == True:
        # evaluate points along curve
        u_i = kwargs.get('u_i', path.u_start())
        u_f = kwargs.get('u_f', path.u_end())
        path_coords = path.list_eval(u_i=u_i, u_f=u_f, n_points=n_points)

        # plot curve
        axes.plot(path_coords[:,0], path_coords[:,1], path_coords[:,2], 
            path_style, label=path_label)
    
    # plot control points if desired
    if show_control_points == True:
        axes.plot(path.P[:,0], path.P[:,1], path.P[:,2], control_point_style, 
            label=control_point_label)
    
    # plot control net if desired
    if show_control_net == True:
        axes.plot(path.P[:,0], path.P[:,1], path.P[:,2], control_net_style, 
            alpha=0.3, label=control_net_label)
    
    # plot knots if desired
    if show_knots == True:
        knots = path.knot_eval()
        axes.plot(knots[:,0], knots[:,1], knots[:,2], knot_style, 
            label=knot_label)
    
    # return axes
    return axes 

def surf_plot_2D(surface, axes=None, show_mesh=True, show_boundary=False, 
    show_control_points=True, show_weights=False, show_control_net=True, 
    show_knots=True, mesh_label='Surface', boundary_label='Boundary', 
    control_point_label='Control Points', control_net_label='Control Net', 
    knot_label='Knots', mesh_colour='black', boundary_colour='black', 
    control_point_style='ro', control_net_colour='blue', knot_style='gx',  
    mesh_alpha=0.1,  boundary_alpha=0.3, control_net_alpha=0.3, N_u=20, N_v=20, 
    **kwargs):
    """
    Creates 3D plot of a surface.
    
    Arguments:
        surface = surface object

    Keyword arguments:
        axes = matplotlib axes
        show_mesh = option to show surface mesh
        show_boundary = option to show boundary of surface
        show_control_points = option to show control points
        show_weights = option to show control point weights
        show_control_net = option to show control net
        show_knots = option to show knots
        mesh_label = label of surface mesh on legend
        boundary_label = label of surface boundary on legend
        control_point_label = label of control points on legend
        control_net_label = label of control net on legend
        knot_label = label of knot on legend
        mesh_colour = colour of surface mesh
        boundary_colour = colour of surface boundary
        control_point_style = matplotlib style of plotted control points
        control_net_colour = colour of control net
        knot_style = matplotlib style of plotted knots
        mesh_alpha = transparency of surface mesh
        boundary_alpha = transparency of surface boundary
        control_net_alpha = transparency of control net
        N_u = number of surface points in u direction
        N_v = number of surface points in v direction
        u_i = initial u coordinate
        u_f = final u coordinate
        v_i = initial v coordinate
        v_f = final v coordinate
    """
    # check if surface is 2D
    dims = np.nan * np.ones((len(surface.P), len(surface.P[0])))
    for i in range(len(surface.P)):
        for j in range(len(surface.P[0])):
            dims[i][j] = len(surface.P[i][j])
    if not np.all(dims == 2):
        raise AttributeError('Not all control points are 2D.')
    del dims

    # initialise axes
    if axes == None:
        axes = plt.axes()
    
    # plot surface mesh if desired
    if show_mesh == True:
        # get plotting end points
        u_i = kwargs.get('u_i', surface.u_start())
        u_f = kwargs.get('u_f', surface.u_end())
        v_i = kwargs.get('v_i', surface.v_start())
        v_f = kwargs.get('v_f', surface.v_end())
        
        # evalute surface points
        surf_coords = surface.list_eval(u_i=u_i, u_f=u_f, N_u=N_u, 
            v_i=v_i, v_f=v_f, N_v=N_v)
        surf_xs, surf_ys = surf_coords[:,:,0], surf_coords[:,:,1]
        
        # create vertical and horizontal grid lines
        v_grid = np.stack((surf_xs, surf_ys), axis=2)
        h_grid = v_grid.transpose(1, 0, 2)
        
        # add to axes
        axes.add_collection(LineCollection(v_grid, color=mesh_colour, 
            alpha=mesh_alpha, label=mesh_label))
        axes.add_collection(LineCollection(h_grid, color=mesh_colour, 
            alpha=mesh_alpha))
    
    # plot hull of surface if desired
    if show_boundary == True:
        # get end-points of surface
        u_i = kwargs.get('u_i', surface.u_start())
        u_f = kwargs.get('u_f', surface.u_end())
        v_i = kwargs.get('v_i', surface.v_start())
        v_f = kwargs.get('v_f', surface.v_end())

        # plot u boundaries
        u_bndry = surface.list_eval(u_i=u_i, u_f=u_f, N_u=2, 
            v_i=v_i, v_f=v_f, N_v=N_v)
        u_bndry_1_xs, u_bndry_1_ys = u_bndry[0,:,0], u_bndry[0,:,1]
        u_bndry_2_xs, u_bndry_2_ys = u_bndry[1,:,0], u_bndry[1,:,1]
        
        axes.plot(u_bndry_1_xs, u_bndry_1_ys, 
            color=boundary_colour, alpha=boundary_alpha, label=boundary_label)
        axes.plot(u_bndry_2_xs, u_bndry_2_ys, color=boundary_colour, 
            alpha=boundary_alpha)

        # plot v boundaries
        v_bndry = surface.list_eval(u_i=u_i, u_f=u_f, N_u=N_u, 
            v_i=v_i, v_f=v_f, N_v=2)
        v_bndry_1_xs, v_bndry_1_ys = v_bndry[:,0,0], v_bndry[:,0,1]
        v_bndry_2_xs, v_bndry_2_ys = v_bndry[:,1,0], v_bndry[:,1,1]
        
        axes.plot(v_bndry_1_xs, v_bndry_1_ys, color=boundary_colour, 
            alpha=boundary_alpha)
        axes.plot(v_bndry_2_xs, v_bndry_2_ys, color=boundary_colour, 
            alpha=boundary_alpha)

    # plot control points if desired
    if show_control_points == True:
        P_flat = surface.P.reshape((len(surface.P)*len(surface.P[0]), 
            len(surface.P[0][0])))
        
        axes.plot(P_flat[:,0], P_flat[:,1], control_point_style, 
            label=control_point_label)

        if show_weights == True:
            for i in range(len(surface.G)):
                for j in range(len(surface.G[0])):
                    weight_label = '$G_{%i, %i} = $' % (i, j)
                    weight_label += f' ${float(surface.G[i][j]):.4}$'
                    axes.annotate(weight_label, (surface.P[i][j][0], 
                        surface.P[i][j][1]), textcoords='offset points', 
                        xytext=(0, 10), ha='center')

    # plot control net if desired
    if show_control_net == True:
        v_net = np.stack((surface.P[:,:,0], surface.P[:,:,1]), axis=2)
        h_net = v_net.transpose(1, 0, 2)
        axes.add_collection(LineCollection(v_net, color=control_net_colour, 
            alpha=control_net_alpha, label=control_net_label))
        axes.add_collection(LineCollection(h_net, color=control_net_colour, 
            alpha=control_net_alpha))

    # plot knots if desired
    if show_knots == True:
        knots = surface.knot_eval()
        knots_flat = knots.reshape((len(surface.U)*len(surface.V), 
            len(surface.P[0][0])))
        
        axes.plot(knots_flat[:,0], knots_flat[:,1], knot_style, 
            label=knot_label)
    
    # return axes
    return axes

def surf_plot_3D(surface, axes=None, show_wireframe=True, show_colourmap=False, 
    show_control_points=True, show_control_net=True, 
    show_knots=True, wireframe_label='Surface', colourmap_label='Surface', 
    control_point_label='Control Points', control_net_label='Control Net', 
    knot_label='Knots', wireframe_colour='black', colourmap='viridis', 
    control_point_style='ro', control_net_style='b-', knot_style='gx',  
    wireframe_alpha=0.4, control_net_alpha=0.3, N_u=20, N_v=20, 
    **kwargs):
    """
    Creates 3D plot of a surface.
    
    Arguments:
        surface = surface object

    Keyword arguments:
        axes = matplotlib axes
        show_wireframe = option to show surface as a wireframe
        show_colourmap = option to show surface as a colourmap
        show_control_points = option to show control points
        show_control_net = option to show control net
        show_knots = option to show knots
        wireframe_label = label of wireframe surface on legend
        colourmap_label = label of colourmap surface on legend
        control_point_label = label of control points on legend
        control_net_label = label of control net on legend
        knot_label = label of knot on legend
        wireframe_colour = colour of surface wireframe
        colourmap = matplotlib colourmap for colourmap surface
        control_point_style = matplotlib style of plotted control points
        control_net_style = matplotlib style of plotted control net
        knot_style = matplotlib style of plotted knots
        wireframe_alpha = transparency of surface wireframe
        control_net_alpha = transparency of control net
        N_u = number of surface points in u direction
        N_v = number of surface points in v direction
        u_i = initial u coordinate
        u_f = final u coordinate
        v_i = initial v coordinate
        v_f = final v coordinate
    """

    # check if surface is 3D
    dims = np.nan * np.ones((len(surface.P), len(surface.P[0])))
    for i in range(len(surface.P)):
        for j in range(len(surface.P[0])):
            dims[i][j] = len(surface.P[i][j])
    if not np.all(dims == 3):
        raise AttributeError('Not all control points are 3D.')
    del dims
    # initialise axes
    if axes == None:
        axes = plt.axes(projection='3d')
    
    # plot wireframe if desired
    if show_wireframe == True:

        # evaluate points along surface over specified mesh
        u_i = kwargs.get('u_i', surface.u_start())
        u_f = kwargs.get('u_f', surface.u_end())
        v_i = kwargs.get('v_i', surface.v_start())
        v_f = kwargs.get('v_f', surface.v_end())
        surf_coords = surface.list_eval(u_i=u_i, u_f=u_f, N_u=N_u, v_i=v_i, 
            v_f=v_f, N_v=N_v)

        axes.plot_wireframe(surf_coords[:,:,0], surf_coords[:,:,1], 
            surf_coords[:,:,2], label=wireframe_label, color=wireframe_colour, 
            alpha=wireframe_alpha)

    # plot colormap surface if desired
    if show_colourmap == True:
        # evaluate points along surface over specified mesh
        u_i = kwargs.get('u_i', surface.u_start())
        u_f = kwargs.get('u_f', surface.u_end())
        v_i = kwargs.get('v_i', surface.v_start())
        v_f = kwargs.get('v_f', surface.v_end())
        surf_coords = surface.list_eval(u_i=u_i, u_f=u_f, N_u=N_u, v_i=v_i, 
            v_f=v_f, N_v=N_v)
        
        a = axes.plot_surface(surf_coords[:,:,0], surf_coords[:,:,1], 
            surf_coords[:,:,2], cmap=colourmap, label=colourmap_label)
        a._facecolors2d=a._facecolors3d
        a._edgecolors2d=a._edgecolors3d

    # plot control points if desired
    if show_control_points == True:
        P_flat = surface.P.reshape((len(surface.P)*len(surface.P[0]), 
            len(surface.P[0][0])))
        
        axes.plot(P_flat[:,0], P_flat[:,1], P_flat[:,2], control_point_style, 
            label=control_point_label)
    
    # plot control net if desired
    if show_control_net == True:
        # plot control net in u direction
        P_trans = surface.P.transpose(1, 0, 2)
        for i in range(len(P_trans)):
            if i != 0:
                axes.plot(P_trans[i][:,0], P_trans[i][:,1], P_trans[i][:,2], 
                    control_net_style, alpha=control_net_alpha)
            else:
                axes.plot(P_trans[i][:,0], P_trans[i][:,1], P_trans[i][:,2], 
                    control_net_style, alpha=control_net_alpha, 
                    label=control_net_label)
        
        # plot control net in v direction
        for i in range(len(surface.P)):
            axes.plot(surface.P[i][:,0], surface.P[i][:,1], surface.P[i][:,2], 
                control_net_style, alpha=control_net_alpha)
        
    # plotting knots if desired
    if show_knots == True:
        knots = surface.knot_eval()
        knots_flat = knots.reshape((len(surface.U)*len(surface.V), 
            len(surface.P[0][0])))
        
        axes.plot(knots_flat[:,0], knots_flat[:,1], knots_flat[:,2], knot_style, 
            label=knot_label)
    
    # return axes
    return axes