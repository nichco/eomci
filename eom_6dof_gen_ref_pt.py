import numpy as np
import csdl

def _EoM(state_vector, mp_obj, total_force, total_moment):
	# Get state
	u = state_vector[0]
	v = state_vector[1]
	w = state_vector[2]
	p = state_vector[3]
	q = state_vector[4]
	r = state_vector[5]
	phi = state_vector[6]
	theta = state_vector[7]
	psi = state_vector[8]
	x = state_vector[9]
	y = state_vector[10]
	z = state_vector[11]
	# Get total forces
	Fx = total_force[0]
	Fy = total_force[1]
	Fz = total_force[2]
	# Get total moments
	Mx = total_moment[0]
	My = total_moment[1]
	Mz = total_moment[2]
	# Get mass properties
	m = mp_obj.mass.magnitude
	I = mp_obj.inertia.return_array()
	Ixx = mp_obj.inertia.Ixx.magnitude
	Iyy = mp_obj.inertia.Iyy.magnitude
	Izz = mp_obj.inertia.Izz.magnitude
	Ixy = mp_obj.inertia.Ixy.magnitude
	Iyz = mp_obj.inertia.Iyz.magnitude
	Ixz = mp_obj.inertia.Ixz.magnitude
	Idot = np.zeros([3, 3])
	# CG offset from reference point
	Rbc = np.array([mp_obj.cg.x.magnitude,
					 mp_obj.cg.y.magnitude,
					 mp_obj.cg.z.magnitude])
	xcg = Rbc[0]
	ycg = Rbc[1]
	zcg = Rbc[2]
	xcgdot = 0
	ycgdot = 0
	zcgdot = 0
	xcgddot = 0
	ycgddot = 0
	zcgddot = 0

	mp_matrix = np.array([
		[m, 0, 0, 0, m * zcg, -m * ycg],
		[0, m, 0, -m * zcg, 0, m * xcg],
		[0, 0, m, m * ycg, -m * xcg, 0],
		[0, -m * zcg, m * ycg, Ixx, Ixy, Ixz],
		[m * zcg, 0, -m * xcg, Ixy, Iyy, Iyz],
		[-m * ycg, m * xcg, 0, Ixz, Iyz, Izz]
	])

	lambdax = Fx + m * (r * v - q * w
						- xcgddot
						- 2 * q * zcgdot + 2 * r * ycgdot
						+ xcg * (q ** 2 + r ** 2) - ycg * p * q - zcg * p * r)
	lambday = Fy + m * (p * w - r * u
						- ycgddot
						- 2 * r * xcgdot + 2 * p * zcgdot
						- xcg * p * q + ycg * (p ** 2 + r ** 2) - zcg * q * r)
	lambdaz = Fz + m * (q * u - p * v
						- zcgddot
						- 2 * p * ycgdot + 2 * q * xcgdot
						- xcg * p * r - ycg * q * r + zcg * (p ** 2 + q ** 2))

	angvel_vec = np.array([[p], [q], [r]])
	angvel_ssym = np.array([[0, -r, q],
							[r, 0, -p],
							[-q, p, 0]])
	Rbc_ssym = np.array([[0, -zcg, ycg],
						 [zcg, 0, -xcg],
						 [-ycg, xcg, 0]])

	moment_vec = np.array([[Mx], [My], [Mz]])
	mu_vec = moment_vec \
			 - np.dot(Idot, angvel_vec) \
			 - np.dot(np.dot(angvel_ssym, I), angvel_vec) \
			 - m * np.dot(np.dot(Rbc_ssym, angvel_ssym), angvel_vec)

	rhs = np.array([lambdax, lambday, lambdaz,
					mu_vec[0][0], mu_vec[1][0], mu_vec[2][0]])

	#accelerations = np.linalg.solve(mp_matrix, rhs)

    # custom implicit operation
    model = csdl.Model()
    a_mat = model.declare_variable('a_mat')
    b_mat = model.declare_variable('b_mat')
    state = model.declare_variable('state')
    residual = csdl.matmat(a_mat, state) - b_mat
    model.register_output('residual', residual)

    solve_quadratic = self.create_implicit_operation(model)
    solve_quadratic.declare_state('state', residual='residual')
    solve_quadratic.nonlinear_solver = NewtonSolver(
        solve_subsystems=False,
        maxiter=100,
        iprint=False,
        )
    solve_quadratic.linear_solver = ScipyKrylov()    
    
    a_mat = self.declare_variable('a_mat', val=mp_matrix)
    b_mat = self.declare_variable('b_mat', val=rhs)
    accelerations = solve_quadratic(a_mat, b_mat)
    # end custom implicit op
    

	du_dt = accelerations[0]
	dv_dt = accelerations[1]
	dw_dt = accelerations[2]
	dp_dt = accelerations[3]
	dq_dt = accelerations[4]
	dr_dt = accelerations[5]
	dphi_dt = p + q * np.sin(phi) * np.tan(theta) + r * np.cos(phi) * np.tan(theta)
	dtheta_dt = q * np.cos(phi) - r * np.sin(phi)
	dpsi_dt = q * np.sin(phi) / np.cos(theta) + r * np.cos(phi) / np.cos(theta)
	dx_dt = u * np.cos(theta) * np.cos(psi) \
			+ v * (np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi)) \
			+ w * (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi))
	dy_dt = u * np.cos(theta) * np.sin(psi) \
			+ v * (np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi)) \
			+ w * (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi))
	dz_dt = -u * np.sin(theta) + v * np.sin(phi) * np.cos(theta) + w * np.cos(phi) * np.cos(theta)

	return np.array([du_dt, dv_dt, dw_dt,
					 dp_dt, dq_dt, dr_dt,
					 dphi_dt, dtheta_dt, dpsi_dt,
					 dx_dt, dy_dt, dz_dt])