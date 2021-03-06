import math
import inspect
import numpy as np
import time
from scipy import odr, optimize
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
	par1, cov1 = lab._fit_affine_yerr(x, y, dy)
	par3, cov3 = lab._fit_affine_yerr(y, x, dx)
	m, q = par3
	par2 = np.array([1/m, -q/m])
	J = np.array([[-1/m**2, 0], [q/m**2, -1/m]])
	cov2 = J.dot(cov3).dot(J.T)
	icov1 = invs(cov1)
	icov2 = invs(cov2)
	cov = invs(icov1 + icov2)
	par = cov.dot(icov1.dot(par1) + icov2.dot(par2))
	return par, cov

def _fit_curve_odr_3(f, x, y, dx, dy, p0, dfdx=None, dfdps=None, dfdpdxs=None, dfdp=None, dfdpdx=None, **kw):
	dy2 = dy**2
	dx2 = dx**2
	delta = 200
	def fun(p):
		deriv2 = dfdx(x, *p)**2
		effd2 = dy2 + deriv2 * dx2
		return np.sqrt((y - f(x, *p))**2 / effd2 + np.log(effd2 / (1 + deriv2)) + delta)
	if not ((dfdps is None or dfdpdxs is None) and (dfdp is None or dfdpdx is None)):
		rt = np.empty((len(y), len(p0)))
		def jac(p):
			sdfdx = dfdx(x, *p)
			sdfdx2 = sdfdx ** 2
			rad = dy2 + sdfdx2 * dx2
			srad = np.sqrt(rad)
			Res = (y - f(x, *p)) / srad
			res = Res * dx2 * sdfdx
			if not (dfdps is None or dfdpdxs is None):
				for i in range(len(p)):
					sdfdpdx = dfdpdxs[i](x, *p)
					rt[:,i] = (sdfdx * (dx2 - dy2) / ((1+sdfdx2) * rad) * sdfdpdx - (dfdps[i](x, *p) * srad + sdfdpdx * res) / rad * Res) / np.sqrt(Res**2 + np.log(rad / (1 + sdfdx2)) + delta)
			else:
				sdfdpdx = dfdpdx(x, *p)
				rt[:] = ((sdfdx * (dx2 - dy2) / ((1+sdfdx2) * rad)).reshape(-1,1) * sdfdpdx - (dfdp(x, *p) * srad.reshape(-1,1) + sdfdpdx * res.reshape(-1,1)) * (Res / rad).reshape(-1,1)) / np.sqrt(Res**2 + np.log(rad / (1 + sdfdx2)) + delta).reshape(-1,1)
			return rt
	else:
		jac = None
	kw.update(_Nonedict(jac=jac))
	result = optimize.least_squares(fun, p0, **kw)
	par = result.x
	_, s, VT = linalg.svd(result.jac, full_matrices=False)
	threshold = np.finfo(float).eps * max(result.jac.shape) * s[0]
	s = s[s > threshold]
	VT = VT[:s.size]
	cov = np.dot(VT.T / s**2, VT)
	return par, cov, result

def _fit_curve_odr_2(f, x, y, dx, dy, p0, dfdx=None, dfdps=None, dfdpdxs=None, dfdp=None, dfdpdx=None, **kw):
	dy2 = dy**2
	dx2 = dx**2
	delta = 200
	def fun(p):
		deriv2 = dfdx(x, *p)**2
		effd2 = dy2 + deriv2 * dx2
		return np.concatenate(((y - f(x, *p)) / np.sqrt(effd2), np.sqrt(np.log(effd2 / (1 + deriv2)) + delta)))
	if not ((dfdps is None or dfdpdxs is None) and (dfdp is None or dfdpdx is None)):
		rt = np.empty((len(y) * 2, len(p0)))
		def jac(p):
			sdfdx = dfdx(x, *p)
			sdfdx2 = sdfdx ** 2
			rad = dy2 + sdfdx2 * dx2
			srad = np.sqrt(rad)
			res = (y - f(x, *p)) * dx2 * sdfdx / srad
			if not (dfdps is None or dfdpdxs is None):
				for i in range(len(p)):
					sdfdpdx = dfdpdxs[i](x, *p)
					rt[:len(y),i] = - (dfdps[i](x, *p) * srad + sdfdpdx * res) / rad
					rt[len(y):,i] = sdfdx * (dx2 - dy2) / ((1+sdfdx2) * rad * np.sqrt(np.log(rad/(1+sdfdx2)) + delta)) * sdfdpdx
			else:
				sdfdpdx = dfdpdx(x, *p)
				rt[:len(y)] = - (dfdp(x, *p) * srad.reshape(-1,1) + sdfdpdx * res.reshape(-1,1)) / rad.reshape(-1,1)
				rt[len(y):] = (sdfdx * (dx2 - dy2) / ((1+sdfdx2) * rad * np.sqrt(np.log(rad/(1+sdfdx2)) + delta))).reshape(-1,1) * sdfdpdx
			return rt
	else:
		jac = None
	kw.update(_Nonedict(jac=jac))
	result = optimize.least_squares(fun, p0, **kw)
	par = result.x
	_, s, VT = linalg.svd(result.jac, full_matrices=False)
	threshold = np.finfo(float).eps * max(result.jac.shape) * s[0]
	s = s[s > threshold]
	VT = VT[:s.size]
	cov = np.dot(VT.T / s**2, VT)
	return par, cov, result

def _fit_curve_odr_2_bis(f, x, y, dx, dy, p0, dfdx=None, dfdps=None, dfdpdxs=None, dfdp=None, dfdpdx=None, **kw):
	import numdifftools as numdiff
	
	dy2 = dy**2
	dx2 = dx**2
	def minusloglikelihood(p):
		sdfdx = dfdx(x, *p)
		sdfdx2 = sdfdx ** 2
		effd2 = dy2 + sdfdx2 * dx2
		res = y - f(x, *p)
		return 1/2 * np.sum(np.log(effd2 / (1 + sdfdx2)) + res**2 / effd2)
	
	def jac(p):
		sdfdx = dfdx(x, *p)
		sdfdx2 = sdfdx ** 2
		effd2 = dy2 + sdfdx2 * dx2
		effd = np.sqrt(effd2)
		res = y - f(x, *p)
		rt = np.empty(len(p))
		for i in range(len(p)):
			sdfdpdx = dfdpdxs[i](x, *p)
			rt[i] = np.sum(sdfdx * (dx2 - dy2) / (1 + sdfdx2) / effd2 * sdfdpdx - (res * dfdps[i](x, *p) * effd2 + res**2 * dx2 * sdfdx * sdfdpdx) / effd2**2)
		
		return rt
	
	result = optimize.minimize(minusloglikelihood, p0, jac=jac, **kw)
	
	par = result.x
	hess = numdiff.Hessian(minusloglikelihood)(par)
	try:
		cov = linalg.inv(hess)
	except linalg.LinAlgError:
		W, V = linalg.eigh(hess)
		cov = np.zeros(hess.shape)
		threshold = 1e-10
		np.fill_diagonal(cov, [(1 / w if abs(w) > threshold else 0) for w in W])
		cov = V.dot(cov).dot(V.T)
	return par, cov, result

def _fit_curve_odr_2_nd(f, x, y, dx, dy, p0, dfdx=None, dfdps=None, dfdpdxs=None, dfdp=None, dfdpdx=None, **kw):
	import numdifftools as numdiff

	dy2 = dy**2
	dx2 = dx**2
	delta = 200
	def fun(p):
		deriv2 = dfdx(x, *p)**2
		effd2 = dy2 + deriv2 * dx2
		return np.concatenate(((y - f(x, *p)) / np.sqrt(effd2), np.sqrt(np.log(effd2 / (1 + deriv2)) + delta)))
	if not ((dfdps is None or dfdpdxs is None) and (dfdp is None or dfdpdx is None)):
		rt = np.empty((len(y) * 2, len(p0)))
		def jac(p):
			sdfdx = dfdx(x, *p)
			sdfdx2 = sdfdx ** 2
			rad = dy2 + sdfdx2 * dx2
			srad = np.sqrt(rad)
			res = (y - f(x, *p)) * dx2 * sdfdx / srad
			if not (dfdps is None or dfdpdxs is None):
				for i in range(len(p)):
					sdfdpdx = dfdpdxs[i](x, *p)
					rt[:len(y),i] = - (dfdps[i](x, *p) * srad + sdfdpdx * res) / rad
					rt[len(y):,i] = sdfdx * (dx2 - dy2) / ((1+sdfdx2) * rad * np.sqrt(np.log(rad/(1+sdfdx2)) + delta)) * sdfdpdx
			else:
				sdfdpdx = dfdpdx(x, *p)
				rt[:len(y)] = - (dfdp(x, *p) * srad.reshape(-1,1) + sdfdpdx * res.reshape(-1,1)) / rad.reshape(-1,1)
				rt[len(y):] = (sdfdx * (dx2 - dy2) / ((1+sdfdx2) * rad * np.sqrt(np.log(rad/(1+sdfdx2)) + delta))).reshape(-1,1) * sdfdpdx
			return rt
	else:
		jac = None
	kw.update(_Nonedict(jac=jac))
	result = optimize.least_squares(fun, p0, **kw)
	par = result.x

	def minusloglikelihood(p):
		sdfdx = dfdx(x, *p)
		sdfdx2 = sdfdx ** 2
		effd2 = dy2 + sdfdx2 * dx2
		res = y - f(x, *p)
		return 1/2 * np.sum(np.log(effd2 / (1 + sdfdx2)) + res**2 / effd2)

	hess = numdiff.Hessian(minusloglikelihood)(par)
	try:
		cov = linalg.inv(hess)
	except linalg.LinAlgError:
		W, V = linalg.eigh(hess)
		cov = np.zeros(hess.shape)
		threshold = 1e-10
		np.fill_diagonal(cov, [(1 / w if abs(w) > threshold else 0) for w in W])
		cov = V.dot(cov).dot(V.T)

	return par, cov, result
