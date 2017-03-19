import numpy as np
import lab
import time
from scipy import stats, linalg, integrate
import vegas
from matplotlib import gridspec, pyplot

# TODO/NOTE
#
# fit_bayes_1
# va esponenziale nel numero di parametri, è ragionevole fino a due parametri
# 1 parametro: un cazzo
# 2 parametri: 1 secondo
# 3 parametri: 2 minuti
# 4 parametri: non ci ho provato, 4 ore?
# l'np.array dentro fint non mi piace, e bisogna capire come scegliere epsrel / epsabs in modo più fino, queste cose richiedono di modificare integrate.nquad.
# integrare in coordinate sferiche?
# si può ancora guadagnare un fattore costante parallelizzando gli integrali, girando un po' le cose dovrebbero essere TUTTI indipendenti a meno di overflow
# bisogna poter dare i priori!
# di sicuro questa funzione non va bene con gli errori sulle x perché lì ogni x è un parametro
#
# bisogna fare un fit_bayes_2 che usi monte carlo
# metodi possibili:
# 0) sampling uniforme ignorante
# 1) sampling gaussiano basato su p0, covp0
# 2) sampling furbo che spara N uniformi, se la varianza è ancora alta divide lungo x_0 e va avanti ricorsivo (ciclando x_1, x_2, etc altrimenti la suddivisione a cubetti porta un andamento esponenziale)
# 3) impacchettare emcee (mi spaventa scegliere in automatico il burn-in)

def diagonalize_cov(cov):
	"""
	returns V, A with A diagonal such that
	V A V.T = cov
	
	other properties:
	V A^(-1) V.T = cov^(-1)
	p' = V.T p
	p'.T A^(-1) p' = p.T cov^(-1) p
	
	cov must be symmetric!
	
	"""
	w, V = linalg.eigh(cov)
	
	A = zeros((len(w), len(w)))
	np.fill_diagonal(A, w)
	
	return V, A

def fit_bayes_1(f, x, y, dy, p0, cov0):
	p0 = asarray(p0)
	cov0 = asarray(cov0)
	
	V, A = diagonalize_cov(cov0)
	dp0 = sqrt(diag(A))
	p0 = V.T.dot(p0)
		
	# 10^(-3/2) because we show errors with "1.5" digits
	# 1/2 to round correctly
	# 1/2 because the are two sources of error, one is sistematic
	# 1/√2 because everything gets multiplied by normalization
	relerr = 10 ** (-3/2) * 1/2 * 1/2 * 1/np.sqrt(2)
	
	def flim(epsrel):
		radius = abs(stats.norm.ppf(epsrel / 2)) + 1
		radius2 = radius ** 2
		def lim_circle(*p):
			x0 = np.sqrt(radius2 - np.sum(np.array(p)**2))
			return (-x0, x0)
		def lim_square(*p):
			return (-radius, radius)
		return [lim_circle] * len(p0)
	fopts = lambda epsrel: dict(epsabs=0, epsrel=epsrel)
	
	idy2 = 1 / dy ** 2
	chi20 = np.sum((y - f(x, *(V.dot(p0))))**2 * idy2)
	def L(*p):
		# must be centered around the maximum
		# also we must be careful with overflow/underflow
		# return np.exp(-0.5 * np.sum(((y - f(x, *p)) / dy)**2))
		# return np.prod(np.exp(-((y - f(x, *p)) / dy)**2 / 2) / (np.sqrt(2 * np.pi) * dy))
		# return np.exp((-np.sum(((y - f(x, *p)) / dy)**2) + len(x) - len(p0)) / 2)
		return np.exp((-np.sum((y - f(x, *(V.dot(p))))**2 * idy2) + chi20) / 2)
	
	# NORMALIZATION
	
	# we change variable such that the function is already with integral about 1 (we hope)
	# and that the horizontal scale is about 1
	def fint(*p):
		p = np.array(p) * dp0 + p0
		return L(*p)
	
	epsrel = 1 / np.sqrt(len(p0)) * relerr * min(dp0 / np.abs(p0)) # quadrature relative errors sum
	lim = flim(epsrel)
	
	start = time.time()
	N, dN, out = integrate.nquad(fint, lim, opts=fopts(epsrel), full_output=True)
	deltat = time.time() - start
	
	print('Normalization = %s, epsrel: %.2g, neval: %d' % (lab.xe(N, dN), epsrel, out['neval']))

	fa = linspace(-4, 4, 1000)
	
	figure(0)
	clf()
	suptitle('Normalization')
	for i in range(len(p0)):
		subplot(len(p0), 1, i + 1)
		p = zeros(len(p0))
		def fplot(pi):
			p[i] = pi
			return fint(*p)
		plot(fa, [fplot(A) for A in fa], label='Time: %.3g s' % (deltat,))
		if i == 0:
			legend()
	
	figure(1)
	clf()
	suptitle('Average and covariance')
	
	# AVERAGE

	par = np.empty(len(p0))
	for i in range(len(par)):
		def fint(*p):
			p = np.array(p) * dp0 + p0
			return p[i] * L(*p)
		epsrel = 1 / np.sqrt(len(p0)) * relerr * dp0[i] / abs(p0[i])
		lim = flim(epsrel)
		start = time.time()
		par[i], err, out = integrate.nquad(fint, lim, opts=fopts(epsrel), full_output=True)
		par[i] /= N
		err /= N
		deltat = time.time() - start
		
		print('Average_%d = %s, epsrel: %.2g, neval: %d' % (i, lab.xe(par[i], err), epsrel, out['neval']))
		
		subplot(len(p0), len(p0) + 1, i * (len(p0) + 1) + 1)
		p = zeros(len(p0))
		def fplot(pi):
			p[i] = pi
			return fint(*p)
		plot(fa, [fplot(A) for A in fa], label='Time: %.3g s' % (deltat,))
		legend()
	
	# COVARIANCE

	cov = np.empty((len(par), len(par)))
	for i in range(len(par)):
		for j in range(i + 1):
			def fint(*p):
				p = np.array(p) * dp0 + par
				return (p[i] - par[i]) * (p[j] - par[j]) * L(*p)
			epsrel = 1 / np.sqrt(len(p0)) * relerr * np.sqrt(2)
			epsabs = 0
			lim = flim(epsrel)
			if i != j:
				epsabs = epsrel * dp0[i] * dp0[j]
				epsrel = 0
			start = time.time()
			cov[i, j], err, out = integrate.nquad(fint, lim, opts=dict(epsrel=epsrel, epsabs=epsabs), full_output=True)
			cov[i, j] /= N
			err /= N
			deltat = time.time() - start
			cov[j, i] = cov[i, j]
			
			print('Covariance_%d,%d = %s, epsrel: %.2g, epsabs: %.2g, neval: %d' % (i, j, lab.xe(cov[i, j], err), epsrel, epsabs, out['neval']))
		
			subplot(len(p0), len(p0) + 1, i * (len(p0) + 1) + j + 2)
			p = zeros(len(p0))
			def fplot(pij, sign):
				p[i] = pij
				p[j] = sign * pij
				return fint(*p)
			plot(fa, [fplot(A, -1) for A in fa], label='Time: %.3g s' % (deltat,))
			legend()
			if i != j:
				subplot(len(p0), len(p0) + 1, j * (len(p0) + 1) + i + 2)
				plot(fa, [fplot(A, 1) for A in fa], label='Time: %.3g s' % (deltat,))
				legend()
	
	par = V.dot(par)
	cov = V.dot(cov).dot(V.T)

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

def mc_integrator_2(f, bounds, epsrel=1e-4, epsabs=1e-4, start_neval=1000, print_info=True, max_cycles=20, target=lambda I: I):
	if print_info:
		print('############### mc_integrator_2 ###############')
		print()
		print('integrand = {}'.format(f))
		print('bounds = {}'.format(bounds))
		print('epsrel = {}'.format(epsrel))
		print('epsabs = {}'.format(epsabs))
	epsrel = np.asarray(epsrel)
	epsabs = np.asarray(epsabs)
	integ = vegas.Integrator(bounds)
	neval = start_neval
	total_neval = 0
	I = None
	tI = None
	for i in range(1, 1 + max_cycles):
		nitn = 10 if i == 1 else 5
		if print_info:
			print()
			print('***** Cycle %d *****' % i)
			print('Integrating with neval=%d, nitn=%d' % (neval, nitn))
		result = integ(f, nitn=nitn, neval=neval)
		Is = np.array(result.itn_results)
		if len(Is.shape) == 1:
			Is = Is.reshape((len(Is), 1))
		new_I = [sum([J / J.var for J in Is[:,j]]) / sum([1 / J.var for J in Is[:,j]]) for j in range(Is.shape[1])]
		chi2 = sum([sum([(J.mean - new_I[j].mean)**2 / J.var for J in Is[:,j]]) for j in range(Is.shape[1])])
		Q = stats.chi2.sf(chi2, (Is.shape[0] - 1) * Is.shape[1])
		if print_info:
			print(result.summary())
		if Q < 0.05:
			if print_info:
				print('Q = %.2g < 0.05, repeating cycle.' % Q)
			continue
		I = [(I[j]/I[j].var + new_I[j]/new_I[j].var) / (1/I[j].var + 1/new_I[j].var) for j in range(Is.shape[1])] if not (I is None) else new_I
		total_neval += neval
		tI = target(I)
		current_error = np.array([Ij.sdev for Ij in tI])
		if len(tI) < len(np.atleast_1d(epsrel)) or len(tI) < len(np.atleast_1d(epsabs)):
			raise ValueError('Length of target values less than length of target errors.')
		target_error = epsrel * np.array([Ij.mean for Ij in tI]) + epsabs
		if print_info:
			tnI = target(new_I)
			print('from this cycle:  I = {}'.format(tnI if len(tnI) > 1 else tnI[0]))
			print('weighted average: I = {}'.format(tI if len(tI) > 1 else tI[0]))
			print('                 DI = {}'.format(current_error))
			print('epsrel * I + epsabs = {}'.format(target_error))
		if all(current_error <= target_error):
			if print_info:
				print('Termination condition DI <= epsrel * I + epsabs satisfied.')
			break
		target_total_neval = int(np.max(np.round((current_error / target_error) ** 2 * total_neval)))
		target_neval = target_total_neval - total_neval
		neval = max(neval, min(neval * 4, target_neval))
	if print_info:
		print()
		print('############# END mc_integrator_2 #############')
	return tI if tI is None or len(tI) > 1 else tI[0]

def fit_bayes_2(f, x, y, dy, p0, cov0, print_info=False, plot_figure=None):
	"""
	use MC integrals
	"""
	if print_info:
		print('################# fit_bayes_2 #################')
		print()
		print('f = {}'.format(f))
		print('x.shape = {}'.format(x.shape))
		print('y.shape = {}'.format(y.shape))
		print('dy.shape = {}'.format(dy.shape))
		print()
		print('Starting estimate of parameters:')
		print(lab.format_par_cov(par0, cov0))
		print()
		
	# TARGET ERRORS FOR RESULTS
	
	# target relative error on computed standard deviations
	# 10^(-3/2) because we show errors with "1.5" digits
	# 1/2.1 to round correctly
	std_relerr = 10 ** (-3/2) * 1/2.1
	
	# target relative error on computed variances
	# 2 because variance = std ** 2
	var_relerr = std_relerr * 2
	
	# target absolute error on computed averages
	# align with error on standard deviations, based on initial estimate
	avg_abserr = std_relerr * np.sqrt(np.diag(cov0))
	
	# target relative error on correlations
	# 1/1000 because they are written like xx.x %
	# 1/2.1 to round correctly
	cor_relerr = 1/1000 * 1/2.1
	
	# target absolute error on covariance matrix
	cov_abserr = np.abs(cov0) * cor_relerr
	np.fill_diagonal(cov_abserr, np.diag(cov0) * var_relerr)
	
	# sigma factor for statistical errors
	sigma = abs(stats.norm.ppf(1e-3 / 2))
	
	if print_info:
		print('Target relative error on variances = %.3g' % var_relerr)
		print('Target relative error on correlations = %.3g' % cor_relerr)
		print('Target absolute error on averages:\n{}'.format(avg_abserr))
		print('Target absolute error on covariance matrix:\n{}'.format(cov_abserr))
		print()
	
	# VARIABLE TRANSFORM AND LIKELIHOOD
	
	# diagonalize starting estimate of covariance
	w, V = linalg.eigh(cov0)
	dp0 = np.sqrt(w)
	p0 = V.T.dot(p0)
	cov_abserr = np.sqrt(np.abs(V.T.dot(cov_abserr ** 2).dot(V)))
	avg_abserr = np.sqrt(np.abs(V.T.dot(avg_abserr ** 2)))
	
	if print_info:
		print('Diagonalized starting estimate of parameters:')
		print(lab.xe(p0, dp0))
		print()
		print('Diagonalized target absolute error on averages:')
		print(avg_abserr)
		print('Diagonalized target absolute error on covariance matrix:')
		print(cov_abserr)
		print()
		
	# change variable: p0 -> 0, dp0 -> 1
	M = dp0
	Q = p0
	dp0 = np.ones(len(p0))
	p0 = np.zeros(len(p0))
	cov_abserr /= np.outer(M, M)
	avg_abserr /= M
	
	if print_info:
		print('Normalized target absolute error on averages:')
		print(avg_abserr)
		print('Normalized target absolute error on covariance matrix:')
		print(cov_abserr)
		print()

	# likelihood (not normalized)
	idy2 = 1 / dy ** 2 # just for efficiency
	chi20 = np.sum((y - f(x, *(V.dot(M * 0 + Q))))**2 * idy2) # initial normalization with L(0) == 1
	def L(p):
		return np.exp((-np.sum((y - f(x, *(V.dot(M * p + Q))))**2 * idy2) + chi20) / 2)
	
	# TARGET ERRORS FOR COMPUTING
	
	# target relative error on variance integrals
	int_var_relerr = np.diag(cov_abserr) / dp0 ** 2 / sigma
	
	# target absolute error on average integrals
	int_avg_abserr = avg_abserr / sigma
	
	# target absolute error on covariance integrals
	int_cov_abserr = np.copy(cov_abserr) / sigma
	np.fill_diagonal(int_cov_abserr, 0)
	
	# INTEGRALS
	
	# integrand: [L, p0 * L, ..., pd * L, p0 * p0 * L, p0 * p1 * L, ..., p0 * pd * L, p1 * p1 * L, ..., pd * pd * L]
	# change of variable: p = 2 * tan(theta), theta in (-pi/2, pi/2)
	bounds = [(-np.pi/2, np.pi/2)] * len(p0)
	idxs = np.triu_indices(len(p0))
	k = 2
	def integrand(theta):
		t = np.tan(theta)
		p = k * t
		l = np.prod(k * (1 + t ** 2)) * L(p)
		return np.concatenate(((l,), p * l, (np.outer(p, p) * l)[idxs]))
	
	# figure showing sections of the integrand
	if not (plot_figure is None):
		plot_figure.set_tight_layout(True)
		plot_figure.clf()
		G = gridspec.GridSpec(len(p0), len(p0) + 2)
		eps = 1e-4
		xs = np.linspace(-np.pi/2 + eps, np.pi/2 - eps, 256)
		
		for i in range(len(p0)):
			axes = plot_figure.add_subplot(G[i, 0])
			theta = np.zeros(len(p0))
			def fplot(theta_i):
				theta[i] = theta_i
				return integrand(theta)[0]
			axes.plot(xs, [fplot(x) for x in xs], '-k', label=R'$L$ along $p_{%d}$' % (i))
			axes.legend(loc=1)

		for i in range(len(p0)):
			axes = plot_figure.add_subplot(G[i, 1])
			theta = np.zeros(len(p0))
			def fplot(theta_i):
				theta[i] = theta_i
				return integrand(theta)[1 + i]
			axes.plot(xs, [fplot(x) for x in xs], '-k', label=R'$p_{%d}\cdot L$ along $p_{%d}$' % (i, i))
			axes.legend(loc=1, fontsize='small')
		
		mat = np.empty(cov0.shape, dtype='uint32')
		mat[idxs] = np.arange(len(idxs[0]))
		for i in range(len(p0)):
			for j in range(i, len(p0)):
				theta = np.zeros(len(p0))
				if i == j:
					axes = plot_figure.add_subplot(G[i, i + 2])
					def fplot(theta_i):
						theta[i] = theta_i
						return integrand(theta)[1 + len(p0) + mat[i, i]]
					axes.plot(xs, [fplot(x) for x in xs], '-k', label=R'$p_{%d}^2\cdot L$ along $p_{%d}$' % (i, i))
					axes.legend(loc=1, fontsize='small')
				else: # i != j
					def fplot_p(theta_ij):
						theta[i] = theta_ij
						theta[j] = theta_ij
						return integrand(theta)[1 + len(p0) + mat[i, j]]
					def fplot_m(theta_ij):
						theta[i] = theta_ij
						theta[j] = -theta_ij
						return integrand(theta)[1 + len(p0) + mat[i, j]]
					axes = plot_figure.add_subplot(G[i, j + 2])
					axes.plot(xs, [fplot_p(x) for x in xs], '-k', label=R'$p_{%d}\cdot p_{%d}\cdot L$ along $p_{%d}=p_{%d}$' % (i, j, i, j))
					axes.legend(loc=1, fontsize='small')
					axes = plot_figure.add_subplot(G[j, i + 2])
					axes.plot(xs, [fplot_m(x) for x in xs], '-k', label=R'$p_{%d}\cdot p_{%d}\cdot L$ along $p_{%d}=-p_{%d}$' % (i, j, i, j))
					axes.legend(loc=1, fontsize='small')
	
	# takes the integration result and computes average and covariance dividing by normalization
	def target(I):
		out1 = [I[j] / I[0] for j in range(1, len(p0) + 1)]
		out2 = [(I[j + len(p0) + 1] - I[idxs[0][j] + 1] * I[idxs[1][j] + 1]) / I[0] for j in range(len(idxs[0]))]
		return out1 + out2
	
	epsrel = np.zeros(cov0.shape)
	epsrel[np.diag_indices(len(p0))] = int_var_relerr
	epsrel = np.concatenate((np.zeros(len(p0)), epsrel[idxs]))
	
	epsabs = np.copy(int_cov_abserr)
	epsabs = np.concatenate((int_avg_abserr, epsabs[idxs]))
	
	I = mc_integrator_2(integrand, bounds, target=target, epsrel=epsrel, epsabs=epsabs, print_info=print_info)
	
	par = [I[i].mean for i in range(len(p0))]
	cov = np.empty(cov0.shape)
	cov[idxs] = [I[i].mean for i in range(len(p0), len(I))]
	cov.T[idxs] = cov[idxs]
	
	if print_info:
		print()
		print('Normalized result:')
		print(lab.format_par_cov(par, cov))
		print()
		
	# INVERSE VARIABLE TRANSFORM
	
	par = V.dot(M * par + Q)
	cov = V.dot(np.outer(M, M) * cov).dot(V.T)
	
	if print_info:
		print('Result:')
		print(lab.format_par_cov(par, cov))
		print()
		print('############### END fit_bayes_2 ###############')
	
	return par, cov

f = lambda x, a, b, c, d: a * x**2 + b*x + c + d*x**3
p0 = (-1, -2, -3, -4)
x = np.linspace(0, 1, 1000)
dy = np.array([.05] * len(x))

y = f(x, *p0) + np.random.randn(len(x)) * dy

par0, cov0 = lab.fit_generic(f, x, y, dy=dy, p0=p0)

fig = pyplot.figure('fitbayes')

par, cov = fit_bayes_2(f, x, y, dy, par0, cov0, print_info=True, plot_figure=fig)

print(lab.format_par_cov(par0, cov0))
print(lab.format_par_cov(par, cov))

pyplot.show()
