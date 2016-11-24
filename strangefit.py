import math
import inspect
import numpy as np
import time
from scipy import odr
from scipy.optimize import curve_fit, leastsq
import lab

def invs(a):
	"""
	Invert a symmetric matrix.
	"""
	ia = np.linalg.inv(a)
	return (ia + ia.T) / 2	

def fit_generic_xyerr4(f, finv, dfdp, dfinvdp, x, y, dx, dy, p0):
	"""
	Fitta y(x) con errori dy, x(y) con dx e fa la media pesata.
	
	Cosa non va:
	Bias parametro: minore di varianza effettiva ma con andamento più brutto.
	Bias errori: le stime non sono indipendenti quindi l'errore è molto
	sottostimato. In effetti non ha senso statistico fare la media pesata
	"""
	def Dfun(p):
		return -dfdp(x, *p) / dy[np.newaxis, :]
	def residual(p):
		return (y - f(x, *p)) / dy
	def Dfun_inv(p):
		return -dfinvdp(y, *p) / dx[np.newaxis, :]
	def residual_inv(p):
		return (x - finv(y, *p)) / dx
	par1, cov1, _, _, _ = leastsq(residual, p0, Dfun=Dfun, col_deriv=True, full_output=True)
	par2, cov2, _, _, _ = leastsq(residual_inv, par1, Dfun=Dfun_inv, col_deriv=True, full_output=True)
	icov1 = invs(cov1)
	icov2 = invs(cov2)
	cov = invs(icov1 + icov2)
	par = cov.dot(icov1.dot(par1) + icov2.dot(par2))
	return par, cov
	
def fit_linear_hoch(x, y, dx, dy):
	"""
	Stessa cosa di fit_generic_xyerr4 ma per le rette.
	"""
	par1, cov1 = _fit_affine_yerr(x, y, dy)
	par3, cov3 = _fit_affine_yerr(y, x, dx)
	m, q = par3
	par2 = np.array([1/m, -q/m])
	J = np.array([[-1/m**2, 0], [q/m**2, -1/m]])
	cov2 = J.dot(cov3).dot(J.T)
	icov1 = invs(cov1)
	icov2 = invs(cov2)
	cov = invs(icov1 + icov2)
	par = cov.dot(icov1.dot(par1) + icov2.dot(par2))
	return par, cov

def fit_linear_ev(x, y, dx=None, dy=None, offset=True, absolute_sigma=True, conv_diff=1e-7, max_cycles=5, print_info=False):
	"""
	Fit y = m * x + q
	
	If offset=False, fit y = m * x

	Parameters
	----------
	x : M-length array
		x data
	y : M-length array
		y data
	dx : M-length array or None
		standard deviation of x
	dy : M-length array or None
		standard deviation of y
		If both dx and dy are None, the fit behaves as if absolute_sigma=False
		and errors were uniform. If only one of dx or dy is None, the fit
		behaves as if it is zero.
	offset : bool
		If True, fit y = m + x + q; else fit y = m * x
	absolute_sigma : bool
		If True, compute standard error on parameters (maximum likelihood
		estimation assuming datapoints are normal). If False, rescale
		errors on parameters to values that would be obtained if the
		chisquare matched the degrees of freedom.
		Simply said: True for physicists, False for engineers
	conv_diff : number
		the difference in terms of standard deviation that
		is considered sufficient for convergence; see notes
	max_cycles : integer
		the maximum number of fits done; see notes.
		If this maximum is reached, an exception is raised.
	print_info : bool
		If True, print information about the fit

	Returns
	-------
	par:
		estimates (m, q)
	cov:
		covariance matrix m,q

	Notes
	-----
	Algorithm: run fit_affine_yerr once ignoring sigmax, then propagate sigmax
	using the formula:
		 sigmay = sqrt(sigmay**2 + (a * sigmax)**2)
	and run fit_affine_yerr again until the differences between two successive
	estimates of the parameters are less than conv_diff times the standard
	deviation of the last estimate.
	"""
	x = np.asarray(x)
	y = np.asarray(y)
	if offset:
		fun_fit = lab._fit_affine_yerr
		fun_fit_dynone = lab._fit_affine_unif_err
		ddof = 2
	else:
		fun_fit = lab._fit_linear_yerr
		fun_fit_dynone = lab._fit_linear_unif_err
		ddof = 1
	if (dy is None) and (dx is None):
		par, cov = fun_fit_dynone(x, y)
		chisq_rid = ((y - par[0]*x - par[1])**2).sum() / (len(x) - ddof)
		cov *= chisq_rid
		return par, cov
	elif (not dy is None) and (dx is None):
		dy = np.asarray(dy)
		if (dy == 0).any():
			raise ValueError('Fit with fixed points not supported')
		par, cov = fun_fit(x, y, dy)
		if not absolute_sigma:
			chisq_rid = (((y - par[0]*x - par[1]) / dy)**2).sum() / (len(x) - ddof)
			cov *= chisq_rid
		return par, cov
	elif (dy is None) and (not dx is None):
		dx = np.asarray(dx)
		if (dx == 0).any():
			raise ValueError('Fit with fixed points not supported')
		par, cov = fun_fit_dynone(x, y)
		if not absolute_sigma:
			chisq_rid = (((y - par[0]*x - par[1]))**2).sum() / (len(x) - ddof)
			cov *= chisq_rid
		dy = 0
	else:
		dx = np.asarray(dx)
		dy = np.asarray(dy)
		dy0 = dy == 0
		if np.logical_and(dx == 0, dy0).any():
			raise ValueError('Fit with fixed points not supported')
		ndy0 = np.logical_not(dy0)
		if ndy0.sum() > ddof:
			par, cov = fun_fit(x[ndy0], y[ndy0], dy[ndy0])
			if not absolute_sigma:
				chisq_rid = (((y - par[0]*x - par[1]) / dy)**2).sum() / (len(x) - ddof)
				cov *= chisq_rid
		else:
			par, cov = fun_fit_dynone(x, y)
			chisq_rid = (((y - par[0]*x - par[1]))**2).sum() / (len(x) - ddof)
			cov *= chisq_rid
	par, cov, cycles = _fit_affine_ev(fun_fit, x, y, dx, dy, par, cov, absolute_sigma, conv_diff, max_cycles)
	if cycles == -1:
		raise RuntimeError('Max cycles %d reached' % max_cycles)
	if print_info:
		print("fit_linear: cycles: %d" % (cycles))
	return par, cov
	
def _fit_affine_ev(fun_fit, x, y, dx, dy, par, cov, absolute_sigma, conv_diff, max_cycles):
	cycles = 1
	while True:
		if cycles >= max_cycles:
			cycles = -1
			break
		dyeff = np.sqrt(dy**2 + (par[0] * dx)**2)
		npar, ncov = fun_fit(x, y, dyeff)
		if not absolute_sigma:
			chisq_rid = (((y - npar[0]*x - npar[1]) / dyeff)**2).sum() / (len(x) - ddof)
			ncov *= chisq_rid
		error = abs(npar - par) / npar
		cerror = abs(ncov - cov) / ncov
		par = npar
		cov = ncov
		cycles += 1
		if (error < conv_diff).all() and (cerror < conv_diff).all():
			break
	return par, cov, cycles
