"""
Utility functions for performing geometric operations.

Author: Reece Otto 14/03/2022
"""
import numpy as np
from math import sin, cos

def valid_coords(coords):
	"""
	Returns True if a valid set of coordinates were input.

	Arguments:
		coords: numpy array containing 3 dimensional coordinates
	"""
	# check if coords is a numpy array
	if type(coords) != np.ndarray:
		raise TypeError('Coordinates must be input as a numpy.ndarray.')

	# check if coords has a valid shape
	if len(coords.shape) != 3:
		raise AssertionError('Coordinate array has invalid shape.')

	# check if coords contains 3 dimensional coordinates
	if coords.shape[2] != 3:
		raise AssertionError('Given coordinates are not three-dimensional.')

def scale(coords, x_scale=1.0, y_scale=1.0, z_scale=1.0):
	"""
	Scales a given coordinate array by given x, y and z scaling factors.

	Arguments:
		coords: numpy array containing 3 dimensional coordinates

	Keyword arguments:
		x_scale: scaling factor for x coordinates
		y_scale: scaling factor for y coordinates
		z_scale: scaling factor for z coordinates
	"""
	# check if coords is a valid coordinate array
	valid_coords(coords)

	# multiply coordinates by scaling factors
	scaled_coords = np.copy(coords)
	for i in range(len(coords)):
		for j in range(len(coords[0])):
			scaled_coords[i][j][0] *= x_scale
			scaled_coords[i][j][1] *= y_scale
			scaled_coords[i][j][2] *= z_scale

	return scaled_coords

def translate(coords, x_shift=0.0, y_shift=0.0, z_shift=0.0):
	"""
	Translates a given coordinate array by given x, y and z shifts.

	Arguments:
		coords: numpy array containing 3 dimensional coordinates

	Keyword arguments:
		x_shift: displacement of x coordinates
		y_shift: displacement of y coordinates
		z_shift: displacement of z coordinates
	"""
	# check if coords is a valid coordinate array
	valid_coords(coords)

	# translate coordinates
	trans_coords = np.copy(coords)
	for i in range(len(coords)):
		for j in range(len(coords[0])):
			trans_coords[i][j][0] += x_shift
			trans_coords[i][j][1] += y_shift
			trans_coords[i][j][2] += z_shift

	return trans_coords

def rotate_x(coords, angle, x_origin=0.0, y_origin=0.0, z_origin=0.0):
	"""
	Rotates a given coordinate array about the x axis.

	Arguments:
		angle: rotation angle in radians

	Keyword arguments:
		x_origin: x coordinate of point of rotation
		y_origin: y coordinate of point of rotation
		z_origin: z coordinate of point of rotation
	"""
	# check if coords is a valid coordinate array
	valid_coords(coords)

	# translate coordinates to new coordinate frame where desired rotation point
	# is the origin
	if x_origin != 0 and y_origin != 0 and z_origin !=0:
		rot_coords = translate(coords, -x_origin, -y_origin, -z_origin)
	else:
		rot_coords = np.copy(coords)

	# construct rotation matrix
	R_x = np.array([[1.0, 0.0, 0.0],
		            [0.0, cos(angle), -sin(angle)],
		            [0.0, sin(angle), cos(angle)]])

	# rotate coordinate array
	for i in range(len(coords)):
		for j in range(len(coords[0])):
			rot_coords[i][j] = np.matmul(R_x, rot_coords[i][j])

	# translate coordinate back to original reference frame
	rot_coords = translate(rot_coords, x_origin, y_origin, z_origin)

	return rot_coords

def rotate_y(coords, angle, x_origin=0.0, y_origin=0.0, z_origin=0.0):
	"""
	Rotates a given coordinate array about the y axis.

	Arguments:
		angle: rotation angle in radians

	Keyword arguments:
		x_origin: x coordinate of point of rotation
		y_origin: y coordinate of point of rotation
		z_origin: z coordinate of point of rotation
	"""
	# check if coords is a valid coordinate array
	valid_coords(coords)

	# translate coordinates to new coordinate frame where desired rotation point
	# is the origin
	if x_origin != 0 and y_origin != 0 and z_origin !=0:
		rot_coords = translate(coords, -x_origin, -y_origin, -z_origin)
	else:
		rot_coords = np.copy(coords)

	# construct rotation matrix
	R_y = np.array([[cos(angle), 0.0, sin(angle)],
		            [0.0, 1.0, 0.0],
		            [-sin(angle), 0, cos(angle)]])

	# rotate coordinate array
	for i in range(len(coords)):
		for j in range(len(coords[0])):
			rot_coords[i][j] = np.matmul(R_y, rot_coords[i][j])

	# translate coordinate back to original reference frame
	rot_coords = translate(rot_coords, x_origin, y_origin, z_origin)

	return rot_coords

def rotate_z(coords, angle, x_origin=0.0, y_origin=0.0, z_origin=0.0):
	"""
	Rotates a given coordinate array about the z axis.

	Arguments:
		angle: rotation angle in radians

	Keyword arguments:
		x_origin: x coordinate of point of rotation
		y_origin: y coordinate of point of rotation
		z_origin: z coordinate of point of rotation
	"""
	# check if coords is a valid coordinate array
	valid_coords(coords)

	# translate coordinates to new coordinate frame where desired rotation point
	# is the origin
	if x_origin != 0 and y_origin != 0 and z_origin !=0:
		rot_coords = translate(coords, -x_origin, -y_origin, -z_origin)
	else:
		rot_coords = np.copy(coords)

	# construct rotation matrix
	R_z = np.array([[cos(angle), -sin(angle), 0.0],
		            [sin(angle), cos(angle), 0.0],
		            [0.0, 0.0 ,1.0]])

	# rotate coordinate array
	for i in range(len(coords)):
		for j in range(len(coords[0])):
			rot_coords[i][j] = np.matmul(R_z, rot_coords[i][j])

	# translate coordinate back to original reference frame
	rot_coords = translate(rot_coords, x_origin, y_origin, z_origin)

	return rot_coords