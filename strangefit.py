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
