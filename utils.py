### package imports

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import circmean, circstd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from tqdm import tqdm

### simulation parameters

PHI = np.deg2rad(180) # azimuth of wavefront source
THETA = np.deg2rad(135) # polar angle of wavefront source
V = 400 * 1e3 # speed of wavefront propagation [m/s]
T0 = 150 # time of wavefront transit of Earth [s]
TAU = 1e-9 # bias in atomic clock induced by wavefront [s]
ERR = 1e-10 # std of natural noise in atomic clocks [s]

### wavefront fitting and fit analysis

def time_of_arrival(r, vx, vy, vz, t0):
	# Computes wavefront time of arrival for satellite position
	# and wavefront parameters.

	x, y, z = r
	return t0 + (x*vx + y*vy + z*vz)/(vx**2 + vy**2 + vz**2)

def process_events(ts, rs):
	# Returns estimated wavefront parameters and covariance matrix.

	rform = tuple(dim for dim in rs.T)
	return curve_fit(time_of_arrival, rform, ts)

def cartesian(rho, phi, theta):
	# Spherical to cartesian coordinate transformation.

	x = rho*np.sin(theta)*np.cos(phi)
	y = rho*np.sin(theta)*np.sin(phi)
	z = rho*np.cos(theta)
	return np.array([x, y, z])

def spherical_fast(v):
	# Vectorized cartesian to spherical coordinate transformation.

	rho = np.sqrt(v[:,0]**2 + v[:,1]**2 + v[:,2]**2)
	phi = np.arctan2(v[:,1], v[:,0])
	theta = np.arctan2(np.sqrt(v[:,0]**2 + v[:,1]**2), v[:,2])
	return np.stack((rho, phi, theta)).T

def spherical_uncertainty(popt, pcov):
	# Convert cartesian standard deviations to spherical standard
	# deviations with Monte Carlo method.

	n_samples = 1000000
	vs = np.random.multivariate_normal(popt, pcov, n_samples)[:,0:3]
	sph = spherical_fast(vs)
	return (np.std(sph[:,0]), circstd(sph[:,1]), circstd(sph[:,2]))

def visualize(rs, v):
	# Plots events in 3D.

	ax = plt.axes(projection='3d')
	ax.scatter3D(0, 0, 0, marker='*')
	ax.scatter3D(rs[:,0], rs[:,1], rs[:,2], color='black', marker='.')
	ax.scatter3D(v[0], v[1], v[2])
	plt.show()

### GNSS and event simulation

def mat_to_dict(filename):
	# Parses .mat file into dictionary.

	mat = loadmat(filename)
	struct_name = filename[:-4]
	mdata = mat[struct_name][0]
	n = mdata.size
	mdtype = mdata.dtype
	m = len(mdtype)

	data = {}
	for i in range(m):
		name = mdtype.names[i]
		data[name] = np.zeros(n)
		for j in range(n):
			data[name][j] = mdata[j][i][0][0]
	return data

def mean_to_ecc(M, e):
	# Converts mean anomaly to eccentric anomaly.

	# initialization for better convergence
	if M > np.pi or (M > -np.pi and M < 0):
		E = M - e
	else:
		E = M + e

	# iteration
	tol = 1e-12
	d = 1
	while abs(d) > tol:
		d = -(E - e*np.sin(E) - M)/(1 - e*np.cos(E))
		E += d
	return E

def eccentric_anomaly(data, i, t):
	# Propagates anomaly in time.

	asqrt = data['Asqrt'][i]
	t_oe = data['Toe'][i]
	dn = data['Delta_n'][i]
	M0 = data['M0'][i]
	e = data['e'][i]

	mu = 3.986005e14
	a = asqrt**2
	n0 = np.sqrt(mu/a**3)	
	t_k = t - t_oe
	n = n0 + dn
	M = M0 + n*t_k
	M = np.fmod(M, 2*np.pi)
	
	return mean_to_ecc(M, e)

def satellite_position(data, i, t):
	# Outputs position of i-th satellite at time t

	e = data['e'][i]
	omega = data['omega'][i]
	c_us = data['Cus'][i]
	c_uc = data['Cuc'][i]
	c_rs = data['Crs'][i]
	c_rc = data['Crc'][i]
	c_is = data['Cis'][i]
	c_ic = data['Cic'][i]
	asqrt = data['Asqrt'][i]
	t_oe = data['Toe'][i]
	i0 = data['i0'][i]
	i_dot = data['IDOT'][i]
	Omega = data['OMEGA'][i]
	Omega_dot = data['OMEGA_DOT'][i]

	E_k = eccentric_anomaly(data, i, t)
	nu_k = np.arctan2(np.sqrt(1 - e**2)*np.sin(E_k), np.cos(E_k) - e)
	phi_k = nu_k + omega
	du_k = c_us*np.sin(2*phi_k) + c_uc*np.cos(2*phi_k)
	dr_k = c_rs*np.sin(2*phi_k) + c_rc*np.cos(2*phi_k)
	di_k = c_is*np.sin(2*phi_k) + c_ic*np.cos(2*phi_k)
	u_k = phi_k + du_k
	a = asqrt**2
	r_k = a*(1 - e*np.cos(E_k)) + dr_k
	t_k = t - t_oe
	i_k = i0 + di_k + i_dot*t_k
	xp_k = r_k*np.cos(u_k)
	yp_k = r_k*np.sin(u_k)
	Omega_dot_e = 0 # 7.2921151467e-5 (EIEF)
	Omega_k = Omega + (Omega_dot - Omega_dot_e)*t_k - Omega_dot_e*t_oe
	x_k = xp_k*np.cos(Omega_k) - yp_k*np.cos(i_k)*np.sin(Omega_k)
	y_k = xp_k*np.sin(Omega_k) + yp_k*np.cos(i_k)*np.cos(Omega_k)
	z_k = yp_k*np.sin(i_k)
	return [x_k, y_k, z_k]

### event handling

def state_generator(t_start, dt):
	# Generator that returns sequential states with sequential calls.
	# Allows for flexible in-the-loop simulation.
	# Memory efficient.

	# ephemeris data
	data = mat_to_dict('eph.mat')
	idx = np.insert(np.arange(30), [7, 19], [30, 54])
	v = -cartesian(V, PHI, THETA)
	t = t_start

	while True:
		# true satellite position
		rs = [satellite_position(data, i, t) for i in idx]
		# local satellite clock bias
		ts = ERR*np.random.randn(len(idx)) # noise
		# event occurrance
		crit = np.dot(v, v)*(t - T0)
		dts = np.array([TAU if np.dot(r, v) < crit else 0 for r in rs])
		ts += dts
		# generator return
		yield t, ts, np.array(rs)
		t += dt

### in-the-loop processing

def detector(t_start, runtime):
	# Monitors diff of satellite time biases and assembles list of events
	# (when diff exceeds detector sensitivity threshold).
	# Fast and memory efficient.

	sensitivity = 7.5e-10
	sample_time = 0.1 # s
	n_steps = int(runtime/sample_time)

	state_iter = state_generator(t_start, sample_time)
	t, ts_old, rs_old = next(state_iter)

	for _ in tqdm(range(n_steps)):
		t, ts_new, rs_new = next(state_iter)
		dts = ts_new - ts_old
		ts_old, rs_old = ts_new, rs_new
		events = np.abs(dts) > sensitivity
		if np.any(events):
			idx = np.where(events)
			for i in idx[0]:
				# average position value over time interval
				r_avg = (rs_old[i,:] + rs_new[i,:])/2
				yield t, r_avg, dts[i]

def analysis(t_start, runtime):
	# Runs detector for set time and estimates wavefront parameters.

	detected = detector(t_start, runtime)
	state = [(t, r, mag) for t, r, mag in detected]
	ts, rs, mags = map(np.array, zip(*state))
	(popt, pcov) = process_events(ts, rs)

	t0 = popt[3]
	std_t0 = np.sqrt(pcov[3, 3])
	v = popt[0:3]
	v_sph = spherical_fast(np.array([-v])).flatten()
	std_v = spherical_uncertainty(popt, pcov)

	# printing
	s1 = u'Time of earth transit t0 = {:.4f} ± {:.4f} s'
	print(s1.format(t0, std_t0))
	s2 = u'  Wavefront velocity |v| = {:.4f} ± {:.4f} km/s'
	print(s2.format(v_sph[0] * 1e-3, std_v[0] * 1e-3))
	s3 = u'        Source azimuth φ = {:.4f} ± {:.4f} deg'
	print(s3.format(np.rad2deg(v_sph[1]), np.rad2deg(std_v[1])))
	s4 = u'      Source elevation θ = {:.4f} ± {:.4f} deg'
	print(s4.format(90 - np.rad2deg(v_sph[2]), np.rad2deg(std_v[2])))

