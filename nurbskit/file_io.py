"""
Functions that export geometric objects to CAD files.

Author: Reece Otto 03/12/2021
"""
from datetime import datetime
from nurbskit.surface import NURBSSurface
import numpy as np
import pyvista as pv
import csv

def holl_string(string):
    """
    Converts a given string to the Hollerith format.

    Arguments:
        string = string to be converted to Hollerith format
    """
    return str(len(string)) + 'H' + string

def iges_prep(file_name='cad_file'):
    """
    Initiliases an IGES file with the 'start' and 'global' entry sections.

    Arguments:
        file_name = name of IGES file to be created
    """

    # 1 - start entry section
    start_lines = ['S'.rjust(73) + '1'.rjust(7) + '\n']
    
    # 2- global entry section
    global_dict = {
        'parameter_delim' : ',',
        'record_delim' : ';',
        'product_ID_sender' : 'NURBS',
        'file_name' : file_name + '.IGES',
        'native_system_ID' : 'nurbskit',
        'preprocessor_version' : 'V0.1.0',
        'int_bits' : 32,
        'single_prec_mag' : 38,
        'single_prec_sig' : 6,
        'double_pre_mag' : 38,
        'double_prec_sig' : 15,
        'product_ID_receiver' : 'NURBS',
        'scale' : 1.0,
        'units_flag' : 2,
        'units_name' : 'MM',
        'max_no_line_grad' : 1,
        'max_line_width_weight' : 0.1,
        'file_date_time' : datetime.now().strftime("%Y%m%d.%H%M%S"),
        'min_res' : 1.0E-06,
        'approx_max_coord' : 0.0,
        'author_name' : 'Reece Otto',
        'author_org' : 'UQ',
        'version_flag' : 11,
        'drafting_standard' : 0,
        'model_date_time' : 'NULL',
        'app_identifier' : 'NULL'
    }
    
    global_lines = []
    global_str = ''
    g = 1
    for i in global_dict:
        if type(global_dict[i]) == str:
            if i == 'app_identifier':
                global_param = holl_string(global_dict[i]) + \
                               global_dict['record_delim']
                global_str += global_param
            else:
                global_param = holl_string(global_dict[i]) + \
                               global_dict['parameter_delim']
                global_str += global_param
        else:
            if i == 'app_identifier':
                global_param = str(global_dict[i]) + \
                               global_dict['record_delim']
                global_str += global_param
            else:
                global_param = str(global_dict[i]) + \
                               global_dict['parameter_delim']
                global_str += global_param

        # ensure global_str doesn't go past column 72 (use 71 just in case)
        if len(global_str) >= 71:
            global_str = global_str[:len(global_str) - len(global_param)]
            l_space = 73 - len(global_str)
            r_space = 81 - len(global_str) - l_space - 1
            global_lines.append(global_str + 'G'.rjust(l_space) + 
                                str(g).rjust(r_space) + '\n')
            global_str = global_param
            g += 1
        if i == 'app_identifier':
            l_space = 73 - len(global_str)
            r_space = 81 - len(global_str) - l_space - 1
            global_lines.append(global_str + 'G'.rjust(l_space) + 
                                str(g).rjust(r_space) + '\n')

    return start_lines, global_lines, global_dict

def nurbs_surf_to_iges(NURBSSurface, file_name='nurbs_surf'):
    """
    Outputs a NURBS surface as an IGES file.

    Arguments:
        NURBSSurface = NURBSSurface object
        file_name = name of IGES file to be created
    """
    start_lines, global_lines, global_dict = iges_prep(file_name = file_name)
    # 4 - parameter entry section (must be established before section 3)
    # flatten control point and weight arrays
    p_str = ''
    g_str = ''
    
    V_str = ''
    for j in range(len(NURBSSurface.P[0])):
        for i in range(len(NURBSSurface.P)):
            g_str += f'{NURBSSurface.G[i][j]:g}' + ','
            p_str += global_dict['parameter_delim'].join(str(NURBSSurface.P[i][j])[1:-1].split()) + \
                     global_dict['parameter_delim']
    
    U_str = ''
    for i in range(len(NURBSSurface.U)):
        U_str += f'{NURBSSurface.U[i]:g}' + global_dict['parameter_delim']
    
    V_str = ''
    for i in range(len(NURBSSurface.V)):
        V_str += f'{NURBSSurface.V[i]:g}' + global_dict['parameter_delim']
    
    param_dict = {
        'entity_no' : 128,
        'upper_ind_sum_u' : int(len(NURBSSurface.P) - 1),
        'upper_ind_sum_v' :  int(len(NURBSSurface.P[0]) - 1),
        'degree_u' : NURBSSurface.p,
        'degree_v' : NURBSSurface.q,
        'closed_u' : NURBSSurface.u_closed(),
        'closed_v' : NURBSSurface.v_closed(),
        'rational' : 0,
        'periodic_u' : NURBSSurface.u_periodic(),
        'periodic_v' : NURBSSurface.v_periodic(),
        'knot_vec_u' : U_str[:-1],
        'knot_vec_v' : V_str[:-1],
        'weights' : g_str[:-1],
        'cntrl_pts' : p_str[:-1],
        'u_start' : f'{NURBSSurface.u_start():g}',
        'u_end' : f'{NURBSSurface.u_end():g}',
        'v_start' : f'{NURBSSurface.v_start():g}',
        'v_end' : f'{NURBSSurface.v_end():g}',
    }
    param_lines = []
    param_str = ''
    p = 1
    
    for i in param_dict:
        if i == 'v_end':
            param = str(param_dict[i]) + global_dict['record_delim']
            param_str += param
        else:
            param = str(param_dict[i]) + global_dict['parameter_delim']
            param_str += param
    
    
    param_str_slice = param_str
    p = 1
    n = 64
    while len(param_str_slice) > n:
        # find the first max length sub-string that ends in the parameter 
        # delimiter but is also less than 71 characters long
        str_71 = param_str_slice[:n]
        if str_71[-1] == global_dict['parameter_delim']:
            param_str_i = str_71
        else:
            str_71_split = str_71.rsplit(global_dict['parameter_delim'], 1)
            param_str_i = str_71_split[0] + global_dict['parameter_delim']
        
        # append parameter number and line number to param_str_i while ensuring
        # they remain in their dedicated column numbers
        l_space = 73 - len(param_str_i)
        r_space = 81 - len(param_str_i) - l_space - 1
        param_str_i += '1P'.rjust(l_space) + str(p).rjust(r_space) + '\n'
        param_lines.append(param_str_i)
        p += 1
        
        # remove str_71 from param_str_slice and prepend str_71_split[1] to
        # param_str_slice if str_71 did not end with parameter delimiter
        if str_71[-1] == global_dict['parameter_delim']:
            param_str_slice = param_str_slice[n:]
        else:
            param_str_slice = str_71_split[1] + param_str_slice[n:]
    
    # add remaining string, parameter number and line number to param_lines
    l_space = 73 - len(param_str_slice)
    r_space = 81 - len(param_str_slice) - l_space - 1
    param_lines.append(param_str_slice + '1P'.rjust(l_space) + 
                       str(p).rjust(r_space) + '\n')
        
    
    #print(param_lines)
    # 3 - directory entry section
    dir_dict = {
        'entity_no1' : 128,
        'param_data' : 1,
        'structure' : 0,
        'line_font_pattern' : 0,
        'level' : 0,
        'view' : 0,
        'trans_matrix' : 0,
        'label_disp_assoc' : 0,
        'status_no' : '00000000',
        'seq_no1' : 'D      1',
        'entity_no11' : 128,
        'line_weight_no' : 0,
        'color_no' : 0,
        'param_line_count' : int(len(param_lines)),
        'form_no' : 0,
        'res1' : '',
        'res2' : '',
        'entity_no' : '',
        'entity_sub_no' : '',
        'seq_no2' : 'D      2'
    }
    
    dir_lines = []
    dir_str = ''
    for i in dir_dict:
        if (i == 'seq_no1') or (i == 'seq_no2'):
            dir_str += str(dir_dict[i]).rjust(8) + '\n'
        else:
            dir_str += str(dir_dict[i]).rjust(8)
    dir_lines.append(dir_str)
    
    # 5 - terminate section
    term_dict = {
        'global' : str(len(global_lines)) + 'G',
        'dir' : '2D',
        'param' : str(len(param_lines)) + 'P'
    }
    term_str = 'S'
    for i in term_dict:
        term_str += term_dict[i].rjust(8)
    
    term_str += 'T'.rjust(73-len(term_str)) + '1'.rjust(7)
    
    f = open(file_name + '.IGES',"w+")
    for line in start_lines:
        f.write(line)
    for line in global_lines:
        f.write(line)
    for line in dir_lines:
        f.write(line)
    for line in param_lines:
        f.write(line)
    f.write(term_str)
    
    f.close()

def surf_to_vtk(Surface, file_name='surface', N_u=100, N_v=100):
    """
    Outputs a surface as an VTK file.
    
    Arguments:
        Surface = Surface object
        file_name = name of IGES file to be created
        N_u = number of mesh points along u direction
        N_v = number of mesh points along v direction
    """
    surf_coords = Surface.discretize(N_u=N_u, N_v=N_v)
    surf_grid = pv.StructuredGrid(surf_coords[:,:,0], surf_coords[:,:,1], 
        surf_coords[:,:,2])
    surf_grid.save(file_name + '.vtk')

P_ROW = 0
Q_ROW = 1
U_ROW = 2
V_ROW = 3
N_PU_ROW = 4
N_PV_ROW = 5
DIM_ROW = 6
FIRST_PT_ROW = 7
    
def import_nurbs_surf(file_name):
    i = 0
    with open(file_name + '.csv') as csv_file:
        nurbs_data = list(csv.reader(csv_file, delimiter=' '))
        for index, row in enumerate(nurbs_data):
            if index == P_ROW:
                p = int(row[0])
            elif index == Q_ROW:
                q = int(row[0])
            elif index == U_ROW:
                U = np.array([float(i) for i in row])
            elif index == V_ROW:
                V = np.array([float(i) for i in row])
            elif index == N_PU_ROW:
                N_Pu = int(row[0])
            elif index == N_PV_ROW:
                N_Pv = int(row[0])
                last_pt_row = N_Pu*N_Pv + FIRST_PT_ROW
            elif index == DIM_ROW:
                dim = int(row[0])
                P = np.zeros((N_Pu, N_Pv, dim))
                G = np.zeros((N_Pu, N_Pv))
            elif index >= FIRST_PT_ROW and index <= last_pt_row:
                j = (index - FIRST_PT_ROW) % N_Pv
                if j == 0 and index != FIRST_PT_ROW:
                    i += 1
                P[i][j] = [float(i) for i in row[:-1]]
                G[i][j] = float(row[-1])

    return NURBSSurface(p=p, q=q, U=U, V=V, P=P, G=G)