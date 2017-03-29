import numpy as np
import lab
import time
from scipy import stats, linalg, integrate, optimize
import vegas, gvar
from matplotlib import gridspec, pyplot
import sympy

# TODO/NOTE
#
# _fit_generic_ml
# jacobiano, max_nfev = 200*, check dx vel dy==0, output
#
# mc_integrator_1
# supportare le stesse funzioni di mc_integrator_2
#
# mc_integrator_2
# aggiungere timeout
# aggiungere condizione di terminazione nonzero
#
# fit_bayes
# aggiungere timeout
# poter usare sia mc_integrator_1 che _2
# aggiungere i priori, ad esempio quando fitto sinusoidi vede anche le armoniche
# gestire mc_integrator che restituisce None
# gestire mc_integrator che passa normalizzazione nulla al target
# print_info se è:
# False, 0: non printare
# True, 1: printa
# n >= 2: passa n - 1 a mc_integrator_*
#
# infine, spostare tutto in lab.py e aggiungere bayes='no','mc-auto','mc-basic','mc-vegas' a fit_generic

def _fit_generic_ml(f, x, y, dx, dy, p0, **kw):
	idy = 1 / dy
	idx = 1 / dx
	def fun(px):
		xstar = px[len(p0):]
		return np.concatenate(((y - f(xstar, *px[:len(p0)])) * idy, (x - xstar) * idx))
	px_scale = np.concatenate((np.ones(len(p0)), dx))
	result = optimize.least_squares(fun, np.concatenate((p0, x)), x_scale=px_scale, **kw)
	par = result.x
	_, s, VT = linalg.svd(result.jac, full_matrices=False)
	threshold = np.finfo(float).eps * max(result.jac.shape) * s[0]
	s = s[s > threshold]
	VT = VT[:s.size]
	cov = np.dot(VT.T / s**2, VT)
	print(result.message)
	return par, cov

def mc_integrator(f_over_dist, dist_sampler, epsrel=1e-4, epsabs=1e-4, start_n=10000, max_bunch=10000000, print_info=True):
	summ = 0.0
	sum2 = 0.0
	i = 0
	n = 0
	while True:
		if i == 0:
			dn = start_n
			if print_info:
				print('################ mc_integrator ################')
				print()
				print('epsrel = %.2g' % epsrel)
				print('epsabs = %.2g' % epsabs)
				print()
				print('***** Cycle %d *****' % i)
				print('Generating start_n = %d samples' % start_n)
		else:
			target_error = epsabs + I * epsrel
			target_n = int((DI / target_error) ** 2 * n)
			dn = min(target_n - n, max_bunch)
			if print_info:
				print()
				print('***** Cycle %d *****' % i)
				print('Estimated necessary samples = %d' % target_n)
				print('Generating %d more samples (max = %d)' % (dn, max_bunch))
		sample = dist_sampler(dn)
		n += dn
		y = f_over_dist(sample)
		summ += np.sum(y, axis=0)
		sum2 += np.sum(y ** 2, axis=0)
		I = summ / n
		DI = np.sqrt((sum2 - summ**2 / n) / (n * (n-1)))
		if print_info:
			print('Result with %d samples:' % n)
			print('I = %s  (I = %g, DI = %g)' % (lab.xe(I, DI), I, DI))
		if all(DI < epsrel * I + epsabs):
			if print_info:
				print('Termination condition DI < epsrel * I + epsabs satisfied.')
				print()
				print('############## END mc_integrator ##############')
			break
		i += 1
	return I, DI

def mc_integrator_2(integrand, bounds, epsrel=1e-4, epsabs=1e-4, start_neval=1000, print_info=True, max_cycles=20, target_result=lambda I: I, target_error=lambda I: I, Q_bound=0.001, **kw):
	if print_info:
		print('############### mc_integrator_2 ###############')
		print()
		print('integrand = {}'.format(f))
		print('bounds[0] = {}'.format(bounds[0]))
		print('epsrel = {}'.format(epsrel))
		print('epsabs = {}'.format(epsabs))
		print('max_cycles = %d' % max_cycles)
		print('Q bound = %.2g' % Q_bound)
	epsrel = np.asarray(epsrel)
	epsabs = np.asarray(epsabs)
	
	integ = vegas.Integrator(bounds)
	
	neval = start_neval
	nevals = []
	tI = None
	I = None
	results = None
	cycle_nitn = 2
	for i in range(1, 1 + max_cycles):
		# INTEGRATION
		nitn = 10 if i == 1 else cycle_nitn
		if print_info:
			print()
			print('***** Cycle %d *****' % i)
			print('Integrating with neval=%d, nitn=%d...' % (neval, nitn))
		result = integ(integrand, nitn=nitn, neval=neval, **kw) # place where things are actually done
		new_results_raw = np.array(result.itn_results)
		
		# APPLY TARGET AND SAVE RESULTS
		if len(new_results_raw.shape) == 3 and new_results_raw.shape[2] == 1:
			new_results_raw = new_results_raw[:,:,0]
		if print_info:
			print(    'itn   raw integrals')
			for j in range(len(new_results_raw)):
				print('%3d   {}'.format(new_results_raw[j]) % (j + len([] if results is None else results) + 1))
			print()
		new_results_0 = target_result(new_results_raw[0])
		new_results = np.empty((len(new_results_raw), len(new_results_0)), dtype=gvar.GVar)
		new_results[0] = new_results_0
		for j in range(1, len(new_results)):
			new_results[j] = target_result(new_results_raw[j])
		results = np.concatenate((results, new_results)) if not (results is None) else new_results
		nevals += [neval] * nitn
		
		# COMPUTE WEIGHTED AVERAGES, Q
		# compute weighted averages and p-values for the last k results for any k.
		wavgs = np.empty(results.shape, dtype=gvar.GVar)
		Qs = np.empty(len(results))
		wavg_nom = 0
		wavg_den = 0
		for j in range(len(results)):
			R = results[-j - 1]
			wavg_nom += np.array(list(map(lambda x: x / x.var, R)))
			wavg_den += np.array(list(map(lambda x: 1 / x.var, R)))
			wavg = wavg_nom / wavg_den
			wavgs[j] = wavg
			chisq = 0
			for k in range(j + 1):
				Rk = results[-k - 1]
				chisq += np.sum(np.fromiter(map(lambda x, y: (x.mean - y.mean)**2 / x.var, Rk, wavg), 'float64', count=len(Rk)))
			Qs[j] = stats.chi2.sf(chisq, j * len(R)) if j != 0 else 0.5
		
		# DECIDE VALUE OF INTEGRAL
		# max last results with p-value in bounds.
		nok = np.sum(np.cumprod(Q_bound < Qs))
		if nok == 1:
			if print_info:
				print('last two results have Q = %.3g, continuing' % Qs[1])
			continue
		I = wavgs[nok - 1]
		tI = target_error(I)
		
		# DECIDE WHAT NEXT
		# check if condition on errors is satisfied, else estimate necessary samples.
		total_neval = np.sum(nevals[-nok:])
		current_error = np.array([Ij.sdev for Ij in tI])
		threshold_error = epsrel * np.abs([Ij.mean for Ij in tI]) + epsabs
		if print_info:
			print('considering last %d/%d results for weighted average (Q = %.3g),' % (nok, len(results), Qs[nok - 1]))
			print('which consist of about %s/%s function evaluations' % tuple(map(lambda x: lab.num2si(x, format='%.3g', space=''), (total_neval, np.sum(nevals)))))
			print('last result:      I = {}'.format(wavgs[0]))
			print('weighted average: I = {}'.format(I))
			print('                 DI = {}'.format(current_error))
			print('epsrel * I + epsabs = {}'.format(threshold_error))
		if all(current_error <= threshold_error):
			if print_info:
				print('Termination condition DI <= epsrel * I + epsabs satisfied.')
			break
		target_total_neval = np.max(np.round((current_error / threshold_error) ** 2 * total_neval))
		target_neval = (target_total_neval - total_neval) / cycle_nitn
		neval = int(max(neval, min(neval * 4, target_neval)))
	
	# RETURN
	if print_info:
		print()
		print('############# END mc_integrator_2 #############')
	return I

def fit_bayes(f, x, y, dx, dy, par0, cov0, relax='x', std_dig=1.5, cor_err=0.01, prob_err=0.05, gamma=1, print_info=False, plot_figure=None):
	"""
	if dx is None, par0-cov0 is the estimate of parameters
	if dx is not None, par0-cov0 is the estimate of [parameters, xs]
	f must be fully vectorized
	"""
	if print_info:
		print('################# fit_bayes #################')
		print()
		print('f = {}'.format(f))
		print('x.shape = {}'.format(x.shape))
		print('y.shape = {}'.format(y.shape))
		print('dy.shape = {}'.format(dy.shape))
		print('dx = None' if dx is None else 'dx.shape = {}'.format(dx.shape))
		print()
		print('Starting estimate of parameters:')
		print(lab.format_par_cov(par0, cov0))
		print()
		
	# TARGET ERRORS FOR RESULTS
	
	# target relative error on computed standard deviations
	# 1/2.1 to round correctly
	std_relerr = 10 ** (-std_dig) * 1/2.1
	
	# target relative error on (1 - correlation)
	# 1/2.1 to round correctly
	onemcor_relerr = cor_err * 1/2.1
	
	# target absolute error on computed averages
	# align with error on standard deviations, based on initial estimate
	dp0 = np.sqrt(np.diag(cov0))
	avg_abserr = std_relerr * dp0
	
	# target relative error on computed variances
	# 2 because variance = std ** 2
	var_relerr = std_relerr * 2
	
	# sigma factor for statistical errors
	sigma = abs(stats.norm.ppf(prob_err / 2))
	
	if print_info:
		print('Target relative error on standard deviations = %.3g' % std_relerr)
		print('Target absolute error on averages:\n{}'.format(avg_abserr))
		print()
	
	# VARIABLE TRANSFORM AND LIKELIHOOD
		
	# diagonalize starting estimate of covariance
	w, V = linalg.eigh(cov0)
	dp0 = np.sqrt(w)
	p0 = V.T.dot(par0)
	
	if print_info:
		print('Diagonalized starting estimate of parameters:')
		print(lab.xe(p0, dp0))
		print()
	
	# change variable: p0 -> C, dp0 -> 1
	# (!) M and Q will be changed by functions, so do not reassign them!
	C = 10
	M = np.copy(dp0)
	Q = p0 - C * M
	dp0 /= M
	p0 = (p0 - Q) / M
	
	# likelihood (not normalized)
	idy2 = 1 / dy ** 2
	vx = x.reshape((1,)+x.shape)
	vy = y.reshape((1,)+y.shape)
	if dx is None:
		chi20 = np.sum((y - f(x, *par0))**2 * idy2) # initial normalization with L(C) == 1
		def L(p):
			return np.exp((-np.sum((vy - f(vx, *np.einsum('ik,jk', V, M.reshape((1,)+M.shape) * p + Q.reshape((1,)+Q.shape)).reshape(p.shape[::-1]+(1,))))**2 * idy2, axis=1) + chi20) / 2)
	else:
		vdx = dx.reshape((1,)+dx.shape)
		chi20 = np.sum((y - f(par0[-len(x):], *par0[:-len(x)]))**2 * idy2) # initial normalization with L(C) == 1
		def L(px):
			tpx = np.einsum('ik,jk', V, M.reshape((1,)+M.shape) * px + Q.reshape((1,)+Q.shape))
			p = tpx[:-len(x)]
			p = p.reshape(p.shape+(1,))
			txstar = tpx[-len(x):].T
			return np.exp((-np.sum((vy - f(txstar, *p))**2 * idy2, axis=1) - np.sum(((txstar - vx) / vdx) ** 2, axis=1) + chi20) / 2)
	
	# INTEGRAND
	
	# integrand: [L, p0 * L, ..., pd * L, p0 * p0 * L, p0 * p1 * L, ..., p0 * pd * L, p1 * p1 * L, ..., pd * pd * L]
	# change of variable: p = C + k * tan(theta), theta in (-pi/2, pi/2)
	idxs = np.triu_indices(len(p0))
	bounds = [(-np.pi/2, np.pi/2)] * len(p0)
	k = 1/2
	@vegas.batchintegrand
	def integrand(theta):
		t = np.tan(theta)
		p = C + k * t
		l = np.prod(k * (1 + t ** 2), axis=1) * L(p)
		return np.concatenate((np.ones(l.shape+(1,)), p, np.einsum('ij,il->ijl', p, p)[(...,)+idxs]), axis=1) * l.reshape(l.shape+(1,))
	
	# figure showing sections of the integrand
	if not (plot_figure is None):
		length = len(p0) if dx is None else len(p0) - len(x)
		plot_figure.clf()
		plot_figure.set_tight_layout(True)
		G = gridspec.GridSpec(length, length + 2)
		subplots = np.empty((length, length + 2), dtype=object)
		eps = 1e-4
		xs = np.linspace(-np.pi/2 + eps, np.pi/2 - eps, 256)
		
		for i in range(length):
			axes = plot_figure.add_subplot(G[i, 0])
			subplots[i,0] = axes
			theta = np.zeros((len(xs), len(p0)))
			def fplot(theta_i):
				theta[:,i] = theta_i
				return integrand(theta)[:,0]
			axes.plot(xs, fplot(xs), '-k', label=R'$L$ along $p_{%d}$' % (i))
			axes.legend(loc=1)

		for i in range(length):
			axes = plot_figure.add_subplot(G[i, 1])
			subplots[i,1] = axes
			theta = np.zeros((len(xs), len(p0)))
			def fplot(theta_i):
				theta[:,i] = theta_i
				return integrand(theta)[:,1 + i]
			axes.plot(xs, fplot(xs), '-k', label=R'$p_{%d}\cdot L$ along $p_{%d}$' % (i, i))
			axes.legend(loc=1, fontsize='small')
		
		mat = np.empty(cov0.shape, dtype='uint32')
		mat[idxs] = np.arange(len(idxs[0]))
		for i in range(length):
			for j in range(i, length):
				theta = np.zeros((len(xs), len(p0)))
				if i == j:
					axes = plot_figure.add_subplot(G[i, i + 2])
					subplots[i,i+2] = axes
					def fplot(theta_i):
						theta[:,i] = theta_i
						return integrand(theta)[:,1 + len(p0) + mat[i, i]]
					axes.plot(xs, fplot(xs), '-k', label=R'$p_{%d}^2\cdot L$ along $p_{%d}$' % (i, i))
					axes.legend(loc=1, fontsize='small')
				else: # i != j
					def fplot_p(theta_ij):
						theta[:,i] = theta_ij
						theta[:,j] = theta_ij
						return integrand(theta)[:,1 + len(p0) + mat[i, j]]
					def fplot_m(theta_ij):
						theta[:,i] = theta_ij
						theta[:,j] = -theta_ij
						return integrand(theta)[:,1 + len(p0) + mat[i, j]]
					axes = plot_figure.add_subplot(G[i, j + 2])
					subplots[i,j+2] = axes
					axes.plot(xs, fplot_p(xs), '-k', label=R'$p_{%d}\cdot p_{%d}\cdot L$ along $p_{%d}=p_{%d}$' % (i, j, i, j))
					axes.legend(loc=1, fontsize='small')
					axes = plot_figure.add_subplot(G[j, i + 2])
					subplots[j,i+2] = axes
					axes.plot(xs, fplot_m(xs), '-k', label=R'$p_{%d}\cdot p_{%d}\cdot L$ along $p_{%d}=-p_{%d}$' % (i, j, i, j))
					axes.legend(loc=1, fontsize='small')
		
	# INTEGRAL
	
	func_mean = np.vectorize(lambda x: x.mean, otypes=['float64'])
	func_sdev = np.vectorize(lambda x: x.sdev, otypes=['float64'])
	
	def matrify(I):
		par = np.asarray(I[:len(p0)])
		cov = np.empty(cov0.shape, dtype=gvar.GVar)
		cov[idxs] = I[len(p0):]
		cov.T[idxs] = cov[idxs]
		return par, cov
	
	# takes the integration result and computes average and covariance dividing by normalization
	def normalize(I):
		par = [I[j] / I[0] for j in range(1, len(p0) + 1)]
		cov_u = [I[j + len(par) + 1] / I[0] - par[idxs[0][j]] * par[idxs[1][j]] for j in range(len(idxs[0]))]
		return matrify(par + cov_u)
	
	# do inverse variable transform
	def target_result(I):
		par, cov = normalize(I)
		par = V.dot(M * par + Q)
		cov = V.dot(np.outer(M, M) * cov).dot(V.T)
		return np.concatenate((par, cov[idxs]))
	
	if isinstance(relax, str):
		relax = [relax]
	if 'x' in relax:
		if dx is None:
			nrelax = np.ones(len(p0), dtype=bool)
		else:
			nrelax = np.arange(len(p0)) < len(p0) - len(x)
	ignore_corr = 'corr' in relax
	if hasattr(relax, '__iter__') and all([isinstance(a, bool) for a in relax]):
		nrelax = ~np.asarray(relax)
	if hasattr(relax, '__getitem__') and 'indexes' in relax:
		nrelax = ~np.asarray(relax['indexes'])
	
	# compute objects on which errors are defined: avg, var, (1-corr)**2
	gamma = 1
	err_np0 = np.sum(nrelax)
	err_idxs = np.diag_indices(err_np0) if 'corr' in relax else np.triu_indices(err_np0)
	nrelax2 = np.outer(nrelax, nrelax)
	def target_error(I):
		par, cov = matrify(I)
		par = par[nrelax]
		cov = cov[nrelax2].reshape(2 * [err_np0])
		sigma = np.sqrt(np.diag(cov))
		var_onemcor2 = (1 - cov / np.outer(sigma, sigma))**2
		np.fill_diagonal(var_onemcor2, np.diag(cov))
		
		if gamma != 1:
			Q[:] = gamma * Q + (1 - gamma) * (V.T.dot(func_mean(par)) - C * M)
			M[:] = gamma * M + (1 - gamma) * np.sqrt(np.diag(V.T.dot(func_mean(cov)).dot(V)))
		
		return np.concatenate((par, var_onemcor2[err_idxs]))

	epsrel = np.ones([err_np0] * 2) * onemcor_relerr / sigma * 2
	np.fill_diagonal(epsrel, var_relerr / sigma)
	epsrel = np.concatenate((np.zeros(err_np0), epsrel[err_idxs]))
	
	epsabs = np.zeros(epsrel.shape)
	epsabs[:err_np0] = avg_abserr[nrelax] / sigma
	
	I = mc_integrator_2(integrand, bounds, target_result=target_result, target_error=target_error, epsrel=epsrel, epsabs=epsabs, print_info=print_info)
	
	# FINALLY, RESULT!
	
	par, cov = matrify(I)
	
	if print_info:
		print()
		# print('Normalized result:')
		# print(lab.format_par_cov(func_mean((V.T.dot(par) - Q) / M), func_mean(V.T.dot(cov).dot(V) / np.outer(M, M))))
		# print('Diagonalized result:')
		# print(lab.format_par_cov(func_mean(V.T.dot(par)), func_mean(V.T.dot(cov).dot(V))))
			
	# plot final variable transform
	if not (plot_figure is None) and gamma != 1:
		for i in range(length):
			axes = subplots[i, 0]
			theta = np.zeros((len(xs), len(p0)))
			def fplot(theta_i):
				theta[:,i] = theta_i
				return integrand(theta)[:,0]
			axes.plot(xs, fplot(xs), '-r')
			axes.legend(loc=1)

		for i in range(length):
			axes = subplots[i, 1]
			theta = np.zeros((len(xs), len(p0)))
			def fplot(theta_i):
				theta[:,i] = theta_i
				return integrand(theta)[:,1 + i]
			axes.plot(xs, fplot(xs), '-r')
			axes.legend(loc=1, fontsize='small')
		
		for i in range(length):
			for j in range(i, length):
				theta = np.zeros((len(xs), len(p0)))
				if i == j:
					axes = subplots[i, i + 2]
					def fplot(theta_i):
						theta[:,i] = theta_i
						return integrand(theta)[:,1 + len(p0) + mat[i, i]]
					axes.plot(xs, fplot(xs), '-r')
				else: # i != j
					def fplot_p(theta_ij):
						theta[:,i] = theta_ij
						theta[:,j] = theta_ij
						return integrand(theta)[:,1 + len(p0) + mat[i, j]]
					def fplot_m(theta_ij):
						theta[:,i] = theta_ij
						theta[:,j] = -theta_ij
						return integrand(theta)[:,1 + len(p0) + mat[i, j]]
					axes = subplots[i, j + 2]
					axes.plot(xs, fplot_p(xs), '-r')
					axes = subplots[j, i + 2]
					axes.plot(xs, fplot_m(xs), '-r')

	dpar = func_sdev(par)
	dcov = func_sdev(cov)
	
	mpar = func_mean(par)
	mcov = func_mean(cov)
		
	if print_info:
		print('Starting estimate was:')
		print(lab.format_par_cov(par0, cov0))
		print('Result:')
		print(lab.format_par_cov(mpar, mcov))
		print('Averages with computing errors:')
		print(par)
		print('Standard deviations with computing errors:')
		sigma = np.sqrt(np.diag(cov))
		print(sigma)
		print('Correlations with computing errors:')
		print(cov / np.outer(sigma, sigma))
		print()
		print('############### END fit_bayes ###############')
	
	return mpar, mcov, dpar, dcov

f_sym = lambda x, a, b: a * x + sympy.sin(b*x)
p0 = (-4, -10)
x0 = np.linspace(0, 1, 5)
dy = np.array([.05] * len(x0)) * 1
dx = np.array([.05] * len(x0)) * 0.4

model = lab.FitModel(f_sym)
f = model.f()

y = f(x0, *p0) + np.random.randn(len(x0)) * dy
x = x0 + np.random.randn(len(x0)) * dx

nitn = 400*(len(x)+len(p0))

# ODR

par0, cov0, out = lab.fit_generic(model, x, y, dx=dx, dy=dy, p0=p0, full_output=True, method='odrpack', print_info=1, maxit=nitn)

# MAXIMUM LIKELIHOOD

PAR, COV = _fit_generic_ml(f, x, y, dx, dy, p0, max_nfev=nitn)
par1 = PAR[:len(p0)]
cov1 = COV[:len(p0),:len(p0)]
X = PAR[len(p0):]

PAR0 = np.concatenate((par0, x + out.delta_x))
COV0 = np.zeros(COV.shape)
COV0[:len(p0),:len(p0)] = cov0
np.fill_diagonal(COV0[len(p0):,len(p0):], dx ** 2)

# BAYESIAN AVERAGE

kwargs = dict(std_dig=1.5, prob_err=0.05, relax=['x', 'corr'])

fig = pyplot.figure('fitbayes_ml')
PARB, COVB, DPARB, DCOVB = fit_bayes(f, x, y, dx, dy, PAR, COV, print_info=True, plot_figure=fig, **kwargs)
parB = PARB[:len(p0)]
XB = PARB[len(p0):]

# RESULTS

print(lab.format_par_cov(PAR0, COV0))
print(lab.format_par_cov(PAR, COV))
# print(lab.format_par_cov(PARA, COVA))
print(lab.format_par_cov(PARB, COVB))
print(p0)

# PLOT OF FITS

fx = np.linspace(min(x), max(x), 512)
fig = pyplot.figure('fitbayes_plot')
fig.clf()
fig.set_tight_layout(True)
axes = fig.add_subplot(111)

# PARENT DISTRIBUTION
axes.plot(fx, f(fx, *p0), '-', color=[.7]*3, linewidth=4, label='parent', zorder=-1)
axes.plot(x0, f(x0, *p0), '.', color=[.7]*3, markersize=16, zorder=-1)
for i in range(len(x0)):
	axes.plot([x0[i], x[i]], [f(x0[i], *p0), y[i]], '--', color=[.7]*3, zorder=-1, linewidth=4)

# SIMULATED DATA
axes.errorbar(x, y, xerr=dx, yerr=dy, fmt=',k', zorder=0, label='data')

# ODR 
axes.plot(fx, f(fx, *par0), '-r', zorder=1, label='odr', linewidth=2)
axes.plot(x + out.delta_x, y + out.delta_y, '.r', zorder=1, markersize=8)
for i in range(len(x)):
	axes.plot([x[i], (x + out.delta_x)[i]], [y[i], (y + out.delta_y)[i]], '--r', zorder=1, linewidth=2)

# LIKELIHOOD
axes.plot(fx, f(fx, *par1), '-b', zorder=2, label='ml', linewidth=1)
axes.plot(X, f(X, *par1), '.b', zorder=2, markersize=4)
for i in range(len(X)):
	axes.plot([x[i], X[i]], [y[i], f(X[i], *par1)], '--b', zorder=2, linewidth=1)

# BAYESIAN
axes.plot(fx, f(fx, *parB), '-y', zorder=4, label='bayes_ml', linewidth=1)
axes.plot(XB, f(XB, *parB), '.y', zorder=4, markersize=4)
for i in range(len(XB)):
	axes.plot([x[i], XB[i]], [y[i], f(XB[i], *parB)], '--y', zorder=4, linewidth=1)

axes.legend(loc=0)
pyplot.show()
