# ********************** IMPORTS ***************************

from __future__ import division, print_function
import math
import inspect
import numpy as np
import time
from scipy import odr, optimize, stats, special, linalg
import os
import sympy
import uncertainties

__all__ = [ # things imported when you do "from lab import *"
	'fit_norm_cov',
	'fit_curve',
	'fit_curve_bootstrap',
	'CurveModel',
	'FitCurveOutput',
	'fit_linear',
	'fit_const_yerr',
	'fit_oversampling',
	'util_mm_er',
	'util_mm_esr',
	'util_mm_esr2',
	'util_mm_list',
	'mme',
	'num2si',
	'num2sup',
	'num2sub',
	'unicode_pm',
	'format_par_cov',
	'xe',
	'xep',
	'util_format',
	'util_timecomp',
	'util_timestr',
	'Eta',
	'nextfilename',
	'sanitizefilename'
]

# __all__ += [ # things for backward compatibility
# 	'curve_fit_patched',
# 	'fit_generic_xyerr',
# 	'fit_generic_xyerr2',
#	'etastart',
#	'etastr'
# ]

__version__ = '2017.05'

# ************************** FIT ***************************

def fit_norm_cov(cov):
	"""
	normalize a square matrix so that the diagonal is 1:
	ncov[i,j] = cov[i,j] / sqrt(cov[i,i] * cov[j,j])

	Parameters
	----------
	cov : (N,N)-shaped array-like
		the matrix to normalize

	Returns
	-------
	ncov : (N,N)-shaped array-like
		the normalized matrix
	"""
	cov = np.copy(np.asarray(cov, dtype='float64'))
	s = np.sqrt(np.diag(cov))
	for i in range(len(s)):
		for j in range(i + 1):
			p = s[i] * s[j]
			if p != 0:
				cov[i, j] /= p
			elif i != j:
				cov[i, j] = np.nan
			cov[j, i] = cov[i, j]
	return cov

def _fit_curve_ev(f, dfdx, x, y, dx, dy, par, cov, absolute_sigma=True, conv_diff=1e-7, max_cycles=5, **kw):
	cycles = 1
	while True:
		if cycles >= max_cycles:
			cycles = -1
			break
		dyeff = np.sqrt(dy**2 + (dfdx(x, *par) * dx)**2)
		rt = optimize.curve_fit(f, x, y, p0=par, sigma=dyeff, absolute_sigma=absolute_sigma, **kw)
		npar, ncov = rt[:2]
		error = abs(npar - par) / npar
		cerror = abs(ncov - cov) / ncov
		par = npar
		cov = ncov
		cycles += 1
		if (error < conv_diff).all() and (cerror < conv_diff).all():
			break
	return par, cov, cycles

class _Nonedict(dict):
	
	def __init__(self, **kw):
		for k in kw:
			if not (kw[k] is None):
				self[k] = kw[k]

def _fit_curve_odr(f, x, y, dx, dy, p0, dfdx=None, dfdps=None, dfdpdxs=None, dfdp=None, dfdpdx=None, **kw):
	dy2 = dy**2
	dx2 = dx**2
	def fun(p):
		return (y - f(x, *p)) / np.sqrt(dy2 + dfdx(x, *p)**2 * dx2)
	if not ((dfdps is None or dfdpdxs is None) and (dfdp is None or dfdpdx is None)):
		rt = np.empty((len(y), len(p0)))
		def jac(p):
			sdfdx = dfdx(x, *p)
			rad = dy2 + sdfdx**2 * dx2
			srad = np.sqrt(rad)
			res = (y - f(x, *p)) * dx2 * sdfdx / srad
			if not (dfdps is None or dfdpdxs is None):
				for i in range(len(p)):
					rt[:,i] = - (dfdps[i](x, *p) * srad + dfdpdxs[i](x, *p) * res) / rad
			else:
				rt[:] = - (dfdp(x, *p) * srad.reshape(-1,1) + dfdpdx(x, *p) * res.reshape(-1,1)) / rad.reshape(-1,1)
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

def _fit_curve_ml(f, x, y, dx, dy, p0, dfdx=None, dfdps=None, dfdp=None, bounds=None, **kw):
	idy = 1 / dy
	idx = 1 / dx
	def fun(px):
		xstar = px[-len(x):]
		return np.concatenate(((y - f(xstar, *px[:len(p0)])) * idy, (x - xstar) * idx))
	if not (dfdx is None) and not (dfdps is None and dfdp is None):
		jacm = np.zeros((len(y) + len(x), len(p0) + len(x)))
		def jac(px):
			p = px[:len(p0)]
			xstar = px[-len(x):]
			if not dfdps is None:
				for i in range(len(p0)):
					jacm[:len(y), i] = -dfdps[i](xstar, *p) * idy
			else:
				jacm[:len(y), :len(p0)] = -dfdp(xstar, *p) * idy
			np.fill_diagonal(jacm[:len(y), -len(x):], -dfdx(xstar, *p) * idy)
			np.fill_diagonal(jacm[-len(x):, -len(x):], -idx)
			return jacm
	else:
		jac = None
	px_scale = np.concatenate((np.ones(len(p0)), dx))
	if not (bounds is None):
		bnds = np.empty((2, len(p0) + len(x)))
		bnds[:,:len(p0)] = np.asarray(bounds).reshape(2,-1)
		bnds[:,-len(x):] = [[-np.inf], [np.inf]]
	else:
		bnds = None
	kw.update(_Nonedict(jac=jac, bounds=bnds))
	result = optimize.least_squares(fun, np.concatenate((p0, x)), x_scale=px_scale, **kw)
	par = result.x
	_, s, VT = linalg.svd(result.jac, full_matrices=False)
	threshold = np.finfo(float).eps * max(result.jac.shape) * s[0]
	s = s[s > threshold]
	VT = VT[:s.size]
	cov = np.dot(VT.T / s**2, VT)
	return par, cov, result

def _asarray(a):
	A = np.asarray(a)
	return a if len(A.shape) == 0 else A

class FitCurveOutput:
	"""
	Object that holds the output of fit_curve.
	
	Parameters
	----------
	par, cov, px, pxcov, datax, datay, fitx, fity, chisq, deltax, deltay,
	method, rawoutput :
		These parameters coincide with members (see their description).
	nump : positive integer or None
		Number of parameters. If px or pxcov are given but not one of datax,
		deltax, datay, deltay, fity, then nump shall be specified.
	check : bool
		If True, perform some consistency checks.
	
	Members
	-------
	par : 1D array
		Estimate of the parameters.
	cov : 2D array
		Covariance matrix of the estimate.
	px : 1D array
		Estimate of parameters, including x (for fits with uncertainties
		along x). The format is:
		[p0, ..., pn, x1, ..., xm]
	pxcov : 2D array
		Covariance matrix of px.
	datax :
		x data. For fits with uncertainties along x, it is a 1D array.
	datay : 1D array
		y data.
	fitx : 1D array
		Fitted xs (for fits with uncertainties along x).
	fity : 1D array
		Fitted ys, i.e. model_function(fitted x, fitted parameters).
	deltax : 1D array
		fitx - datax
	deltay : 1D array
		fity - datay
	chisq : non-negative number
		A chisquare statistics, that is a statistics (i.e. function of
		data) that has a chisquare distribution under the assumption that
		the model is true.
	chisq_dof : positive integer
		Degrees of freedom of chisq.
	chisq_pvalue : non-negative number
		Survival function at chisq, i.e. integral of the chisquare
		distribution with chisq_dof degrees of freedom from chisq to
		infinity.
	method : string
		Fitting algorithm used to generate this result, see fit_curve.
	rawoutput : object
		Object returned by fitting function of lower level than fit_curve,
		if any (tipically a minimizer), see fit_curve.
	"""
	
	def __init__(self, par=None, cov=None, px=None, pxcov=None, nump=None, datax=None, datay=None, fitx=None, fity=None, deltax=None, deltay=None, chisq=None, method=None, rawoutput=None, check=True):
		if not (px is None and pxcov is None):
			A = np.array([datax, deltax, datay, deltay, fity], dtype=object)
			Anone = np.array([a is None for a in A], dtype=bool)
			if nump is None and not all(Anone):
				nump = -len(A[~Anone][0])
			else:
				raise ValueError("You should specify nump")
		
		if not (px is None):
			self.px = np.asarray(px)
			self.par = self.px[:nump]
			self.fitx = self.px[nump:]
			if check:
				assert par is None
				assert fitx is None
		else:
			self.par = _asarray(par)
			self.fitx = _asarray(fitx)
			
		if not (pxcov is None):
			self.pxcov = np.asarray(pxcov)
			self.cov = self.pxcov[:nump,:nump]
			if check:
				assert cov is None
		else:
			self.cov = _asarray(cov)
		
		if hasattr(self, 'px') and hasattr(self, 'pxcov'):
			self.upx = uncertainties.correlated_values(self.px, self.pxcov)
		if not (self.par is None) and not (self.cov is None):
			self.upar = uncertainties.correlated_values(self.par, self.cov)
		
		if deltax is None and not (datax is None) and not (self.fitx is None):
			self.datax = np.asarray(datax)
			self.deltax = self.fitx - self.datax
		elif not (deltax is None) and datax is None and not (self.fitx is None):
			self.deltax = np.asarray(deltax)
			self.datax = self.fitx - self.deltax
		elif not (deltax is None) and not (datax is None) and self.fitx is None:
			self.datax = np.asarray(datax)
			self.deltax = np.asarray(deltax)
			self.fitx = self.datax + self.deltax
		else:
			self.datax = _asarray(datax)
			self.deltax = _asarray(deltax)
		if check and np.sum([a is None for a in [self.fitx, self.datax, self.deltax]]) == 0:
			assert np.allclose(self.fitx, self.datax + self.deltax)
			
		if deltay is None and not (datay is None) and not (fity is None):
			self.datay = np.asarray(datay)
			self.fity = np.asarray(fity)
			self.deltay = self.fity - self.datay
		elif not (deltay is None) and datay is None and not (fity is None):
			self.deltay = np.asarray(deltay)
			self.fity = np.asarray(fity)
			self.datay = self.fity - self.deltay
		elif not (deltay is None) and not (datay is None) and fity is None:
			self.datay = np.asarray(datay)
			self.deltay = np.asarray(deltay)
			self.fity = self.datay + self.deltay
		else:
			self.datay = _asarray(datay)
			self.deltay = _asarray(deltay)
			self.fity = _asarray(fity)
		if check and np.sum([a is None for a in [self.fity, self.datay, self.deltay]]) == 0:
			assert np.allclose(self.fity, self.datay + self.deltay)
		
		self.chisq = chisq
		if not (chisq is None):
			A = np.array([self.datax, self.fitx, self.deltax, self.datay, self.fity, self.deltay], dtype=object)
			Alen = np.array([hasattr(a, '__len__') for a in A], dtype=bool)
			if any(Alen) and not (self.par is None):
				self.chisq_dof = len(A[Alen][0]) - len(self.par)
				self.chisq_pvalue = stats.chi2.sf(self.chisq, self.chisq_dof)
		
		self.rawoutput = rawoutput
		
		self.method = method

class CurveModel:
	"""
	Object to specify a curve model for fit_curve, in the form:
	y = f(x, *par).
	
	Parameters
	----------
	f : function
		A function with signature f(x, *par). Returns the y coordinates
		corresponding to the x coordinates given in the array x for a curve
		parametrized by the arguments *par.
	symb : bool
		If True, derivatives of f respect to x and *par are obtained as
		needed with sympy. In this case, f must accept sympy variables as
		arguments.
	dfdx : function, optional
		A function with the same signature as f. Returns the derivative of
		f respect to x.
	dfdp : function, optional
		A function with the same signature as f. Returns the derivatives of
		f respect to *par, in the form of a 2D array where the first index
		runs along the datapoints and the second along the parameters.
	dfdpdx : function, optional
		A function with the same signature as f. Returns the cross
		derivatives of f respect to x and *par, in the form of a 2D array
		where the first index runs along the datapoints and the second
		along the parameters.
	
	Methods
	-------
	latex
	f
	f_odrpack
	dfdx
	dfdx_odrpack
	dfdps
	dfdp
	dfdp_odrpack
	dfdp_curve_fit
	dfdpdx
	"""

	def __init__(self, f, symb=False, dfdx=None, dfdp=None, dfdpdx=None):
		if symb:
			args = inspect.getargspec(f).args
			xsym = sympy.symbols('x', real=True)
			psym = [sympy.symbols('p%s' % num2sub(i), real=True) for i in range(len(args) - 1)]
			syms = [xsym] + psym
			self._dfdx = sympy.lambdify(syms, f(*syms).diff(xsym), "numpy")
			self._dfdps = [sympy.lambdify(syms, f(*syms).diff(p), "numpy") for p in psym]
			self._dfdpdxs = [sympy.lambdify(syms, f(*syms).diff(xsym).diff(p), "numpy") for p in psym]
			self._f = sympy.lambdify(syms, f(*syms), "numpy")
			self._f_sym = f
			self._repr = 'CurveModel(y = %s)' % (str(f(*syms)).replace('**', '^').replace('*', '·'))
			self._symb = True
		else:
			self._dfdx = dfdx
			self._dfdp = dfdp
			self._dfdpdx = dfdpdx
			self._dfdps = None
			self._dfdpdxs = None
			self._f = f
			self._repr = 'CurveModel(y = {})'.format(f)
			self._symb = False
	
	def __repr__(self):
		return self._repr
	
	def latex(self):
		"""
		Returns
		-------
		s : string
			LaTeX representation of the object.
		"""
		if self._symb:
			args = inspect.getargspec(self._f_sym).args
			xsym = sympy.symbols('x', real=True)
			psym = [sympy.symbols('p_{%d}' % i, real=True) for i in range(len(args) - 1)]
			syms = [xsym] + psym
			return sympy.latex(self._f_sym(*syms))
		else:
			return '\\mathtt{%s}' % format(self._f)
		
	def f(self):
		"""
		Returns
		-------
		f : function
			Model function. If the object was initialized with symb=False,
			it is the f given at initialization; if symb=True, it is a
			"numpyfication" of the symbolic function.
		"""
		return self._f

	def f_odrpack(self, length):
		"""
		Wraps the model function to use the format of scipy.odr.
		
		Parameters
		----------
		length : positive integer
			Number of datapoints the function will be used with.
		
		Returns
		-------
		fcn : function
			Model function with signature fcn(B, x), where B corresponds to
			*par.
		"""
		rt = np.empty(length)
		def f_p(B, x):
			rt[:] = self._f(x, *B)
			return rt
		return f_p

	def dfdx(self):
		"""
		Returns
		-------
		dfdx : function
			Derivative of model function respect to x.
		"""
		return self._dfdx

	def dfdx_odrpack(self, length):
		"""
		Wraps the derivative of model function respect to x to use the
		format of scipy.odr.
		
		Parameters
		----------
		length : positive integer
			Number of datapoints the function will be used with.
		
		Returns
		-------
		jacd : function
			Derivative of model function respect to x with signature
			jacd(B, x), where B corresponds to *par.
		"""
		if self._dfdx is None:
			return None
		else:
			rt = np.empty(length)
			def f_p(B, x):
				rt[:] = self._dfdx(x, *B)
				return rt
			return f_p

	def dfdps(self):
		"""
		Returns
		-------
		dfdps : list of functions or None
			List of derivatives of model function respect to parameters. It
			is available only if the object was initialized with symb=True.
		"""
		return self._dfdps
	
	def dfdp(self, length=None):
		"""
		Returns derivative of model function respect to parameters.
		
		Parameters
		----------
		length : positive integer or None
			Number of datapoints the function will be used with. If the
			model was initialized with symb=False it is not necessary.
		
		Returns
		-------
		dfdp : function or None
			Derivative of model function respect to parameters. Returns a
			2D array where the first index runs along the datapoints and
			the second along the parameters. If symb=False and dfdp was not
			given at initialization, it is None.
		"""
		if self._symb:
			rt = np.empty((length, len(self._dfdps)))
			def f_p(*args):
				for i in range(len(self._dfdps)):
					rt[:,i] = self._dfdps[i](*args)
				return rt
			return f_p
		else:
			return self._dfdp

	def dfdp_odrpack(self, length=None):
		"""
		Wraps the derivative of model function respect to parameters to use
		the format of scipy.odr.
		
		Parameters
		----------
		length : positive integer or None
			Number of datapoints the function will be used with. If the
			model was initialized with symb=False it is not necessary.
		
		Returns
		-------
		jacb : function
			Derivative of model function respect to parameters, with
			signature jacb(B, x), where B corresponds to *par. Returns a 2D
			array where the first index runs along the parameters and the
			second along the datapoints. If symb=False and dfdp was not
			given at initialization, it is None.
		"""
		if self._symb:
			rt = np.empty((len(self._dfdps), length))
			def f_p(B, x):
				for i in range(len(self._dfdps)):
					rt[i] = self._dfdps[i](x, *B)
				return rt
			return f_p
		elif not (self._dfdp is None):
			def f_p(B, x):
				return self._dfdp(x, *B).T
			return f_p
		else:
			return None

	def dfdp_curve_fit(self, length=None):
		"""
		Wraps the derivative of model function respect to parameters to use
		the format of scipy.optimize.curve_fit.
		
		Parameters
		----------
		length : positive integer or None
			Number of datapoints the function will be used with. If the
			model was initialized with symb=False it is not necessary.
		
		Returns
		-------
		jacb : function
			Derivative of model function respect to parameters. Returns a
			2D array where the first index runs along the datapoints and
			the second along the parameters. If symb=False and dfdp was not
			given at initialization, it is None.
		"""
		if self._symb:
			rt = np.empty((len(self._dfdps), length))
			def f_p(*args):
				for i in range(len(self._dfdps)):
					rt[i] = self._dfdps[i](*args)
				return rt.T
			return f_p
		else:
			return self._dfdp

	def dfdpdxs(self):
		"""
		Returns
		-------
		dfdpdxs : list of functions or None
			List of derivatives of model function respect to x and
			parameters. It is available only if the object was initialized
			with symb=True.
		"""
		return self._dfdpdxs
	
	def dfdpdx(self, length=None):
		"""
		Returns derivative of model function respect to x and parameters.
		
		Parameters
		----------
		length : positive integer or None
			Number of datapoints the function will be used with. If the
			model was initialized with symb=False it is not necessary.
		
		Returns
		-------
		dfdp : function or None
			Derivative of model function respect to parameters. Returns a
			2D array where the first index runs along the datapoints and
			the second along the parameters. If symb=False and dfdpdx was
			not given at initialization, it is None.
		"""
		if self._symb:
			rt = np.empty((length, len(self._dfdpdxs)))
			def f_p(*args):
				for i in range(len(self._dfdpdxs)):
					rt[:,i] = self._dfdpdxs[i](*args)
				return rt
			return f_p
		else:
			return self._dfdpdx

def _apply_pfree(f, pfree, p0):
	if not (f is None) and not all(pfree):
		n_par = np.copy(p0)
		def F(x, *par):
			n_par[pfree] = par
			return f(x, *n_par)
		return F
	else:
		return f

def _apply_pfree_par_cov(par, cov, pfree, p0):
	if not all(pfree):
		n_par = np.empty(len(p0), dtype=par.dtype)
		n_par[pfree] = par
		n_par[~pfree] = p0
		n_cov = np.zeros([len(p0)] * 2, dtype=cov.dtype)
		n_cov[np.outer(pfree, pfree)] = cov
		return n_par, n_cov
	else:
		return par, cov

def fit_curve(f, x, y, dx=None, dy=None, p0=None, pfix=None, bounds=None, absolute_sigma=True, method='auto', print_info=0, full_output=True, check=True, **kw):
	"""
	Fit a curve in the form:
	y = f(x, *par)
	finding a "best estimate" for *par.
	
	Parameters
	----------
	f : callable or CurveModel
		If callable: a function with signature f(x, *par), which is used to
		initialize a CurveModel. If CurveModel: it is used directly. See
		CurveModel for details.
	x : 1D array
		x data.
	y : 1D array
		y data.
	dx : 1D array or None
		Uncertainties of x data.
	dy : 1D array or None
		Uncertainties of y data.
	p0 : 1D array
		Initial estimate of *par. Must be specified.
	pfix : None or 1D array either of integers or bools
		Specify which parameters to held fixed to the initial value given
		in p0. If None: all parameters free; if array of integers: the
		integers specify indexes of parameters to fix; if array of bools:
		must have the same shape as *par, a True will mean the
		corresponding parameter is fixed, a False that it is free. Returned
		uncertainties on fixed parameters will be zero.
	bounds : None or 2D array
		Specify bounds inside which parameters are to be searched, in the
		form:
		[[min p0, min p1, ...], [max p0, max p1, ...]]
		+-infinity can be used. None means +-infinity for all parameters.
	absolute_sigma : bool
		If False, multiply estimate of covariance matrix of estimate of
		*par by a factor such that it is as if the uncertainties dx and/or
		dy where scaled by a common factor to get a chisquare statistics
		matching its degrees of freedom.
	method : string, one of 'auto', 'odrpack', 'linodr', 'ml', 'wleastsq',
	'leastsq', 'ev'
		Fitting algorithm to use. If 'auto', choose automatically. See
		below for a description of each algorithm.
	print_info : integer
		Regulate diagnostics printed by the function. If less than or equal
		to 0, print nothing. Positive values mean an increasing amount of
		information; you can safely pass huge values to set the maximum
		level possible.
	full_output : bool
		If False, return less information.
	check : bool
		If False, avoid some consistency checks.
	
	Keyword arguments
	-----------------
	Keyword arguments are passed to the lower-level fitting routine, which
		depends on the method used. See description of methods below.
	
	Fitting methods
	---------------
	All the methods perform a maximum-likelihood fit assuming normal (i.e.
	gaussian) uncertainties.
	'auto' :
		Choose automatically an appropriate method, based on the values of
		dx, dy, bounds given.
	'odrpack' :
		Use ODRPACK through the wrapper scipy.odr. It supports
		uncertainties on both x and y or only on y, and is stable and fast,
		but it does not compute an estimate of the covariance between
		datapoints and parameters in the first case. Bounds are not
		supported. Keyword arguments are passed to scipy.odr.ODR.
	'linodr' :
		Simplifies the computation using a formula that is exact only if
		the model is a straight line; it will be reasonable if at each
		datapoint the radius of curvature is greater enough than the
		uncertainties. the approximation works better if greater
		uncertainties are on y (greater uncertainty on y means that at a
		datapoint the ratio dy/dx is greater than the derivative of the
		curve). It supports uncertainties on at least one of x and y.
		Individual uncertainties may be zero, provided on each datapoint at
		least one of dx and dy is not None or zero. It may become unstable
		if the given CurveModel does not provide a dfdx, case in which a
		single step forward derivative estimation is used. The step can be
		specified with the keyword argument 'diff_step'. Keyword arguments
		(including 'diff_step') are passed to scipy.optimize.least_squares.
	'ml' :
		Supports uncertainties only on both x and y and provides an
		estimate of the covariance between datapoints and parameters.
		Keyword arguments are passed to scipy.optimize.least_squares.
	'leastsq' :
		Ignore given uncertainties, put unitary uncertainties on y and
		apply absolute_sigma=False independently of the value given.
		Keyword arguments are passed to scipy.optimize.curve_fit.
	'wleastsq' :
		Supports no uncertainties or only on y. In the first case behave as
		'leastsq', but do not impose absolute_sigma. Keyword arguments are
		passed to scipy.optimize.curve_fit.
	'ev' :
		Supports uncertainties on x and y or only on y. Works well under
		the same assumptions of linodr, but tipically worse; the same
		considerations on dfdx apply. The algorithm is to repeat the
		procedure of 'wleastsq' many times, using the estimated parameters
		at an iteration to propagate the uncertainties on x to
		uncertainties on y for the next iteration. The keyword argument
		'max_cycles' set a limit on the number of iterations; an exception
		is raised if the limit is surpassed; 'conv_diff' set the relative
		difference between successive estimates (both values and
		covariance) that stops the cycle; 'diff_step' is used as in
		'linodr'. Other keyword arguments are passed to
		scipy.optimize.curve_fit. The output object has a member 'cycles'
		which is the number of cycles done.
	
	Returns
	-------
	out : FitCurveOutput
		Object containing at least the members par and cov, which are the
		estimate of *par and its estimated covariance matrix. If
		full_output=False, these may be the sole members. See
		FitCurveOutput for details.
	
	See also
	--------
	scipy.optimize.curve_fit
	"""
	
	if print_info >= 1:
		print('################ fit_curve ################')
		print()
	
	##### MODEL #####

	if isinstance(f, CurveModel):
		model = f
		if print_info >= 1:
			print('Model given: {}'.format(model))
			print()
	else:
		model = CurveModel(f, symb=False)
		if print_info >= 1:
			print('Model created from f: {}'.format(model))
			print()
	
	if p0 is None:
		raise ValueError("p0 must be specified")
	p0 = np.atleast_1d(p0)
	if not (pfix is None):
		pfix = np.asarray(pfix)
		if np.issubdtype('bool', pfix.dtype):
			pfree = ~pfix
		elif np.issubdtype('int', pfix.dtype) or np.issubdtype('uint', pfix.dtype):
			pfree = np.ones(len(p0), dtype=bool)
			pfree[pfix] = False
	else:
		pfree = np.ones(len(p0), dtype=bool)
	
	if not (bounds is None):
		bounds = np.asarray(bounds).reshape(2,-1)[:,pfree]
	else:
		bounds = np.array([[-np.inf] * len(p0), [np.inf] * len(p0)])[:,pfree]

	##### METHOD #####
	
	if method == 'auto':
		if (dy is None) and not (dx is None):
			method = 'linodr' # only linodr supports errors only along x
		elif dy is None:
			method = 'leastsq'
		elif bounds is None or (all(bounds[0] == -np.inf) and all(bounds[1] == np.inf)):
			method = 'odrpack' # generally good method but does not support bounds
		elif dx is None:
			method = 'wleastsq'
		else:
			method = 'ml' # much slower than odrpack and linodr, but supports bounds and is more correct than linodr
		if print_info >= 1:
			print('Method chosen automatically: "%s"' % method)
			print()
	elif print_info >= 1:
		print('Method: "%s"' % method)
		print()
	
	##### ODRPACK #####
	
	if method == 'odrpack':
		fcn = model.f_odrpack(len(x))
		fjacb = model.dfdp_odrpack(len(x))
		fjacd = model.dfdx_odrpack(len(x))
		
		M = odr.Model(fcn, fjacb=fjacb, fjacd=fjacd)
		data = odr.RealData(x, y, sx=dx, sy=dy)
		ODR = odr.ODR(data, M, beta0=p0, ifixb=pfree, **kw)
		if dx is None:
			ODR.set_job(fit_type=2)
		ip_init = max(0, min(print_info - 1, 2))
		ip_final = ip_init
		ip_iter = max(0, min(print_info - 2, 2))
		ODR.set_iprint(init=ip_init, iter=ip_iter, final=ip_final, iter_step=10)
		output = ODR.run()
		par = output.beta
		cov = output.cov_beta
		
		if full_output or not absolute_sigma:
			if not (dx is None) and not (dy is None):
				chisq = np.sum((output.eps / np.asarray(dy))**2 + (output.delta / np.asarray(dx))**2)
			elif dx is None and not (dy is None):
				chisq = np.sum((output.eps / np.asarray(dy))**2)
			elif dx is None and dy is None:
				chisq = np.sum(output.eps ** 2)
		if not absolute_sigma:
			cov *= chisq / (len(x) - len(par))
		if full_output:
			if dx is None:
				out = FitCurveOutput(par=par, cov=cov, chisq=chisq, deltay=output.eps, datax=x, datay=y, fity=output.y, rawoutput=output, method=method, check=False)
			else:
				out = FitCurveOutput(par=par, cov=cov, chisq=chisq, deltax=output.delta, deltay=output.eps, datax=x, datay=y, fitx=output.xplus, fity=output.y, rawoutput=output, method=method, check=False)
			# (!) check=False because ODRPACK may return slightly inconsistent fity, deltay; the problem is in ODRPACK itself, not in the wrapper. Anyway, inconsistencies are reasonable, so we just look away.
		else:
			out = FitCurveOutput(par=par, cov=cov, check=check)
		if print_info >= 1:
			if print_info > 1:
				print()
			else:
				print(output.stopreason)
			print('Result:')
			print(format_par_cov(par, cov))

	##### LINEARIZED ODR #####

	elif method == 'linodr':
		f = _apply_pfree(model.f(), pfree, p0)
		dfdx = _apply_pfree(model.dfdx(), pfree, p0)
		dfdps = _apply_pfree(model.dfdps(), pfree, p0)
		dfdpdxs = _apply_pfree(model.dfdpdxs(), pfree, p0)
		dfdp = _apply_pfree(model.dfdp(len(x)), pfree, p0)
		dfdpdx = _apply_pfree(model.dfdpdx(len(x)), pfree, p0)
		
		if dfdx is None:
			diff_step = kw.get('diff_step', np.finfo('float64').eps * 65536)
			def dfdx(x, *p):
				h = x * diff_step + diff_step
				return (f(x + h, *p) - f(x, *p)) / h
		
		x = _asarray(x)
		y = np.asarray(y)
		dx = _asarray(dx)
		dy = _asarray(dy)
		if dx is None:
			dx = 0
		if dy is None:
			dy = 0
			
		verbosity = max(0, min(print_info - 1, 2))
		
		par, cov, output = _fit_curve_odr(f, x, y, dx, dy, p0[pfree], dfdx=dfdx, dfdps=dfdps, dfdpdxs=dfdpdxs, dfdp=dfdp, dfdpdx=dfdpdx, verbose=verbosity, bounds=bounds, **kw)
		
		if full_output or not absolute_sigma:
			deriv = dfdx(x, *par)
			err2 = dy ** 2   +   deriv ** 2  *  dx ** 2
			chisq = np.sum((y - f(x, *par))**2 / err2)
		if not absolute_sigma:
			cov *= chisq / (len(x) - len(par))
		if full_output:
			fact = (y - f(x, *par)) / err2
			deltax = fact * deriv * dx**2
			deltay = -fact * dy**2
			par, cov = _apply_pfree_par_cov(par, cov, pfree, p0)
			out = FitCurveOutput(par=par, cov=cov, chisq=chisq, deltax=deltax, deltay=deltay, datax=x, datay=y, method=method, rawoutput=output, check=check)
		else:
			par, cov = _apply_pfree_par_cov(par, cov, pfree, p0)
			out = FitCurveOutput(par=par, cov=cov, check=check)
		
		if print_info >= 1:
			if print_info > 1:
				print()
			else:
				print(output.message)
			print('Result:')
			print(format_par_cov(par, cov))
	
	##### FULL-FEATURE MAXIMUM LIKELIHOOD #####
	
	elif method == 'ml':
		f = _apply_pfree(model.f(), pfree, p0)
		dfdx = _apply_pfree(model.dfdx(), pfree, p0)
		dfdps = _apply_pfree(model.dfdps(), pfree, p0)
		dfdp = _apply_pfree(model.dfdp(len(x)), pfree, p0)
		
		x = np.asarray(x)
		y = np.asarray(y)
		dx = np.asarray(dx)
		dy = np.asarray(dy)
		
		verbosity = max(0, min(print_info - 1, 2))
		
		px, pxcov, output = _fit_curve_ml(f, x, y, dx, dy, p0[pfree], dfdx=dfdx, dfdps=dfdps, dfdp=dfdp, verbose=verbosity, bounds=bounds, **kw)
		
		if full_output or not absolute_sigma:
			par = px[:len(p0[pfree])]
			xstar = px[-len(x):]
			chisq = np.sum(((y - f(xstar, *par)) / dy)**2) + np.sum(((xstar - x) / dx) ** 2)
		if not absolute_sigma:
			pxcov *= chisq / (len(y) - len(par))
		if full_output:
			px, pxcov = _apply_pfree_par_cov(px, pxcov, np.concatenate((pfree, np.ones(len(x), dtype=bool))), p0)
			out = FitCurveOutput(px=px, pxcov=pxcov, datax=x, datay=y, chisq=chisq, method=method, rawoutput=output, check=check)
		else:
			px, pxcov = _apply_pfree_par_cov(px, pxcov, np.concatenate((pfree, np.ones(len(x), dtype=bool))), p0)
			out = FitCurveOutput(px=px, pxcov=pxcov, nump=len(p0), check=check)
		
		if print_info >= 1:
			if print_info > 1:
				print()
			else:
				print(output.message)
			print('Result:')
			print(format_par_cov(px[:len(p0)], pxcov[:len(p0),:len(p0)]))
	
	##### EFFECTIVE VARIANCE #####
	
	elif method == 'ev':
		f = _apply_pfree(model.f(), pfree, p0)
		dfdx = _apply_pfree(model.dfdx(), pfree, p0)
		jac = _apply_pfree(model.dfdp_curve_fit(len(x)), pfree, p0)
		conv_diff = kw.pop('conv_diff', 1e-7)
		max_cycles = kw.pop('max_cycles', 5)
		
		if dfdx is None:
			diff_step = kw.get('diff_step', np.finfo('float64').eps * 65536)
			def dfdx(x, *p):
				h = x * diff_step + diff_step
				return (f(x + h, *p) - f(x, *p)) / h
		
		x = _asarray(x)
		y = np.asarray(y)
		dx = _asarray(dx)
		dy = np.asarray(dy)
		if dx is None:
			dx = 0
		
		par, cov = optimize.curve_fit(f, x, y, p0=p0[pfree], absolute_sigma=absolute_sigma, jac=jac, bounds=bounds, **kw)
		par, cov, cycles = _fit_curve_ev(f, dfdx, x, y, dx, dy, par, cov, absolute_sigma=absolute_sigma, conv_diff=conv_diff, max_cycles=max_cycles, jac=jac, bounds=bounds, **kw)
		
		if cycles == -1:
			raise RuntimeError('Maximum number (%d) of fit cycles reached' % max_cycles)
	
		if full_output:
			deriv = dfdx(x, *par)
			err2 = dy ** 2   +   deriv ** 2  *  dx ** 2
			chisq = np.sum((y - f(x, *par))**2 / err2)
			fact = (y - f(x, *par)) / err2
			deltax = fact * deriv * dx**2
			deltay = -fact * dy**2
			par, cov = _apply_pfree_par_cov(par, cov, pfree, p0)
			out = FitCurveOutput(par=par, cov=cov, datax=x, datay=y, chisq=chisq, method=method, check=check, deltax=deltax, deltay=deltay)
			out.cycles = cycles
		else:
			par, cov = _apply_pfree_par_cov(par, cov, pfree, p0)
			out = FitCurveOutput(par=par, cov=cov, check=check)
	
		if print_info >= 1:
			print('Cycles: %d' % cycles)
			print('Result:')
			print(format_par_cov(par, cov))
	
	##### WEIGHTED LEAST SQUARES #####
	
	elif method == 'wleastsq':
		f = _apply_pfree(model.f(), pfree, p0)
		jac = _apply_pfree(model.dfdp_curve_fit(len(x)), pfree, p0)
		
		par, cov = optimize.curve_fit(f, x, y, sigma=dy, p0=p0[pfree], absolute_sigma=absolute_sigma, jac=jac, bounds=bounds, **kw)
		
		if full_output:
			x = _asarray(x)
			y = np.asarray(y)
			dy = np.asarray(dy)
			deltay = y - f(x, *par)
			chisq = np.sum((deltay / dy) ** 2)
			par, cov = _apply_pfree_par_cov(par, cov, pfree, p0)
			out = FitCurveOutput(par=par, cov=cov, chisq=chisq, deltay=deltay, datax=x, datay=y, method=method, check=check)
		else:
			par, cov = _apply_pfree_par_cov(par, cov, pfree, p0)
			out = FitCurveOutput(par=par, cov=cov, check=check)
	
		if print_info >= 1:
			print('Result:')
			print(format_par_cov(par, cov))
	
	##### LEAST SQUARES #####
	
	elif method == 'leastsq':
		f = _apply_pfree(model.f(), pfree, p0)
		jac = _apply_pfree(model.dfdp_curve_fit(len(x)), pfree, p0)
		
		par, cov = optimize.curve_fit(f, x, y, p0=p0[pfree], absolute_sigma=False, jac=jac, bounds=bounds, **kw)

		if full_output:
			x = _asarray(x)
			y = np.asarray(y)
			deltay = y - f(x, *par)
			chisq = np.sum(deltay ** 2)
			par, cov = _apply_pfree_par_cov(par, cov, pfree, p0)
			out = FitCurveOutput(par=par, cov=cov, chisq=chisq, deltay=deltay, datax=x, datay=y, method=method, check=check)
		else:
			par, cov = _apply_pfree_par_cov(par, cov, pfree, p0)
			out = FitCurveOutput(par=par, cov=cov, check=check)

		if print_info >= 1:
			print('Result:')
			print(format_par_cov(par, cov))
	
	else:
		raise KeyError(method)
	
	# RETURN

	if print_info >= 1:
		print()
		print('############## END fit_curve ##############')
	
	return out

class FitCurveBootstrapOutput:
	
	def __init__(self):
		pass

def fit_curve_bootstrap(f, xmean, dxs=None, dys=None, p0s=None, mcn=1000, method='auto', plot=dict(), eta=False, **kw):
	"""
	Perform a bootstrap, i.e. given a curve model and datapoints with
	uncertainties, generates random displacement to the data with normal
	distribution and standard deviations equal to the uncertainties many
	times, and each time fit the randomly displaced data. Results from fits
	are averaged and optionally graphicated as histograms and scatter
	plots. This is tipically used to verify if the fit works well.
	
	The average is a weighted average, weights are inverses of the
	covariance matrices estimated by the fit.
	
	The arguments are not a single set of uncertainties and parameters:
	dxs, dys and p0s are arrays of arrays and the bootstrap will be
	repeated for each possible combination. This is because this function
	was originally written to study how the fit behaves augmenting
	uncertainties on x and to verify simmetries of parameters; a plot of
	results vs. parameter value or average uncertainty on x or y can be
	generated.
	
	Parameters
	----------
	f : callable of CurveModel
		As the first argument of fit_curve.
	xmean : 1D array
		Central x data. Central y data will be generated with f(xmean,
		*par).
	dxs : 2D array
		Uncertainties on x. The second index runs along datapoints, the
		first along "datasets".
	dys : 2D array
		Uncertainties on y. The second index runs along datapoints, the
		first along datasets. The bootstrap will be repeated for all
		combinations of datasets in dxs and dys.
	p0s : list of arrays
		Values of parameters, in the format:
		[[values for p0], [values for p1], ...]
		The bootstrap will be repeated for all combinations of parameters
		values.
	mcn : positive integer
		"Monte Carlo number": the number of times the fit will be repeated
		for each combination of datasets and parameters.
	method : string
		Fitting method; see fit_curve.
	plot : dictionary
		Argument regulating which plots to draw. The keywords read from the
		dictionary are 'single', 'vsp0', 'vsds', which shall be bools and
		respectively mean: draw a summary plot for each bootstrap; draw a
		plot of results vs. parameter value; draw a plot of results vs.
		uncertainties.
	eta : bool
		If True, every 5 seconds print an estimate of the remaining time.
	
	Keyword arguments
	-----------------
	Keyword arguments are passed to fit_curve.
	
	Returns
	-------
	out : object
		An object with the following members defined:
		fp : array with shape (len(dxs), len(dys), len(p0s[0]),
		len(p0s[1]), ..., len(p0s))
		cp : array with shape (len(dxs), len(dys), len(p0s[0]),
		len(p0s[1]), ..., len(p0s), len(p0s))
			Weighted average of fits results for each bootstrap. fp
			contains the estimated values and cp the estimated covariance
			matrices.
		plotout : dictionary
			Dictionary with the same keywords as those read from the plot
			argument; 'single' contains a list of matplotlib figures, one
			for each bootstrap; 'vsp0' and 'vsds' contain a matplotlib
			figure each. Figures have meaningful titles.
	"""

	n = len(xmean) # number of points
	
	# model
	if isinstance(f, CurveModel):
		model = f
	else:
		model = CurveModel(f, symb=False)
	f = model.f()
	fstr = str(model)
	flatex = model.latex()

	# initialize output arrays
	p0shape = [len(p0) for p0 in p0s]
	fp = np.empty([len(dxs), len(dys)] + p0shape + [len(p0s)]) # fitted parameters (mean over MC)
	cp = np.empty([len(dxs), len(dys)] + p0shape + 2 * [len(p0s)]) # fitted parameters mean covariance matrices
	chisq = np.empty(mcn) # chisquares from 1 MC run
	pars = np.empty((mcn, len(p0s))) # parameters from 1 MC run
	covs = np.empty((mcn, len(p0s), len(p0s))) # covariance matrices from 1 MC run
	times = np.empty(mcn) # execution times from 1 MC run

	def plot_text(string, loc=2, ax=None, **kw):
		locs = [
			[],
			[.95, .95, 'right', 'top'],
			[.05, .95, 'left', 'top']
		]
		loc = locs[loc]
		ax.text(loc[0], loc[1], string, horizontalalignment=loc[2], verticalalignment=loc[3], transform=ax.transAxes, **kw)
	
	if eta:		
		etaobj = Eta()
	plots_single = []
	for ll in range(len(dys)):
		dy = dys[ll]
		for l in range(len(dxs)):
			dx = dxs[l]
			for K in np.ndindex(*p0shape):
				p0 = [p0s[i][K[i]] for i in range(len(K))]
			
				# generate mean data
				ymean = f(xmean, *p0)
			
				# run fits
				for i in range(mcn):
					
					if eta:
						# compute progress
						progress = l + len(dxs) * ll
						for j in range(len(K)):
							progress *= p0shape[j]
							progress += K[j]
						progress *= mcn
						progress += i
						progress /= len(dxs) * len(dys) * np.prod(p0shape) * mcn
						etaobj.etaprint(progress)
				
					# generate data
					deltax = stats.norm.rvs(size=n)
					x = xmean + dx * deltax
					deltay = stats.norm.rvs(size=n)
					y = ymean + dy * deltay
				
					# fit
					start = time.time()
					out = fit_curve(model, x, y, dx, dy, p0=p0, method=method, **_Nonedict(max_cycles=10 if method == 'ev' else None), **kw)
					end = time.time()
				
					# save results
					if plot.get('single', False):
						times[i] = end - start
						chisq[i] = out.chisq
					pars[i] = out.par
					covs[i] = out.cov
			
				# save results
				icovs = np.empty(covs.shape)
				I = np.eye(len(p0))
				for i in range(len(icovs)):
					icovs[i] = linalg.solve(covs[i], I, assume_a='pos')
				pc = linalg.solve(icovs.sum(axis=0), I, assume_a='pos')
				wpar = np.empty(pars.shape)
				for i in range(len(wpar)):
					wpar[i] = icovs[i].dot(pars[i])
				pm = pc.dot(wpar.sum(axis=0))
				pm = pm[:len(p0)]
				pc = pc[:len(p0), :len(p0)]
				ps = np.sqrt(np.diag(pc))

				fp[(l, ll) + K] = pm
				cp[(l, ll) + K] = pc
			
				if plot.get('single', False):
					from matplotlib import pyplot as plt
				
					prho = pc / np.outer(ps, ps)

					pdist = (pm - np.array(p0)) / ps
					pdistc = (np.outer(pm - np.array(p0), pm - np.array(p0)) - pc) / np.sqrt(pc**2 + np.outer(ps, ps)**2)
					chidist = (chisq.mean() - (n-len(p0))) / chisq.std(ddof=1) * np.sqrt(len(chisq))
				
					pvalue = stats.kstest(chisq, 'chi2', (n-len(p0),))[1]
							
					maxscatter = 1000
					histkw = dict(
						bins=int(np.sqrt(min(mcn, 1000))),
						color=(.9,.9,.9),
						edgecolor=(.7, .7, .7)
					)
					if len(p0) == 1:
						rows = 2
						cols = 2
					elif len(p0) == 2:
						rows = 3
						cols = 3
					else:
						rows = 1 + len(p0)
						cols = len(p0)
				
					fig = plt.figure(figsize=(4*cols, 2.3*rows))
					fig.clf()
					fig.set_tight_layout(True)
					fig.canvas.set_window_title('%s, method “%s”' % (fstr, method))
				
					# histogram of parameter; diagonal
					for i in range(len(p0)):					
						ax = fig.add_subplot(rows, cols, 1 + i * (1 + cols))
						ax.set_title("$(p_{%d}'-{p}_{%d})/\sigma_{%d}$" % (i, i, i))
						S = (pars[:,i] - p0[i]) / np.sqrt(covs[:,i,i])
						ax.hist(S, **histkw)
						plot_text("$p_%d = $%g\n$\\bar{p}_{%d}-p_{%d} = $%.2g $\\bar{\sigma}_{%d}$" % (i, p0[i], i, i, pdist[i], i), ax=ax)
					
					# histogram of covariance; lower triangle
					for i in range(len(p0)):
						for j in range(i):
							ax = fig.add_subplot(rows, cols, 1 + i * cols + j)
							ax.set_title("$((p_{%d}'-p_{%d})\cdot(p_{%d}'-p_{%d})-\sigma_{%d%d})/\sqrt{\sigma_{%d%d}^2+\sigma_{%d}^2\sigma_{%d}^2}$" % (i, i, j, j, i, j, i, j, i, j))
							C = ((pars[:,i] - p0[i]) * (pars[:,j] - p0[j]) - covs[:,i,j]) / np.sqrt(covs[:,i,j]**2 + covs[:,i,i]*covs[:,j,j])
							ax.hist(C, **histkw)
							plot_text("$\\bar{\\rho}_{%d%d} = $%.2g\n$(\\bar{p}_{%d}-p_{%d})\cdot(\\bar{p}_{%d}-p_{%d})-\\bar{\sigma}_{%d%d} = $%.2g $\sqrt{\\bar{\sigma}_{%d%d}^2+\\bar{\sigma}_{%d}^2\\bar{\sigma}_{%d}^2}$" % (i, j, prho[i,j], i, i, j, j, i, j, pdistc[i,j], i, j, i, j), ax=ax)
				
					# scatter plot of pairs of parameters; upper triangle
					for i in range(len(p0)):
						for j in range(i + 1, len(p0)):
							ax = fig.add_subplot(rows, cols, 1 + i * cols + j)
							ax.set_title("$(p_{%d}'-p_{%d})/\sigma_{%d}$, $(p_{%d}'-p_{%d})/\sigma_{%d}$" % (i, i, i, j, j, j))
							X = (pars[:, i] - p0[i]) / np.sqrt(covs[:,i,i])
							Y = (pars[:, j] - p0[j]) / np.sqrt(covs[:,j,j])
							if len(X) > maxscatter:
								X = X[::int(ceil(len(X) / maxscatter))]
								Y = Y[::int(ceil(len(Y) / maxscatter))]
							ax.plot(X, Y, '.k', markersize=3, alpha=0.35)
							ax.grid()
				
					# histogram of chisquare; last row column 1
					ax = fig.add_subplot(rows, cols, 1 + cols * (rows - 1))
					ax.set_title('$\chi^2$')
					plot_text('K.S. test p-value = %.2g %%\n$\mathrm{dof}=n-{\#}p = $%d\n$N\cdot(\\bar{\chi}^2 - \mathrm{dof}) = $%.2g $\sqrt{2\cdot\mathrm{dof}}$' % (100*pvalue, n-len(p0), chidist), loc=1, ax=ax)
					ax.hist(chisq, **histkw)

					# histogram of execution time; last row column 2
					ax = fig.add_subplot(rows, cols, 2 + cols * (rows - 1))
					ax.set_title('time')
					plot_text('Average time = %ss' % num2si(times.mean(), format='%.3g'), loc=1, ax=ax)
					ax.hist(times, **histkw)
					ax.ticklabel_format(style='sci', scilimits=(-2,2))

					# example data; last row column 3 (or first row last column)
					ax = fig.add_subplot(rows, cols, (3 + cols * (rows - 1)) if len(p0) >= 2 else cols)
					ax.set_title('Example fit')
					fx = np.linspace(min(xmean), max(xmean), 1000)
					ax.plot(fx, f(fx, *p0), '-', color='lightgray', linewidth=5, label='$y=%s$' % flatex, zorder=1)
					ax.errorbar(x, y, dy, dx, fmt=',k', capsize=0, label='Data', zorder=2)
					ax.plot(fx, f(fx, *pars[-1,:len(p0)]), 'r-', linewidth=1, label='Fit', zorder=3, alpha=1)
					# plot_text('$y=%s$\n$y=%s$' % (flatex, sympy.latex(fsym(xsym, *p0))), fontsize=20, ax=ax)
					ax.ticklabel_format(style='sci', axis='both', scilimits=(-3,3))
					ax.legend(loc=0, fontsize='small')
				
					# save figure
					plots_single.append(fig)

	if plot.get('vsp0', False):
		from matplotlib import pyplot as plt
		fig = plt.figure(figsize=(10,7))
		fig.clf()
		fig.set_tight_layout(True)
		fig.canvas.set_window_title('%s, method “%s”, parameter biases' % (fstr, method))
		for i in range(len(p0s)): # p_i = fitted
			for j in range(len(p0s)): # p_j = true
				ax = fig.add_subplot(len(p0s), len(p0s), 1 + i * len(p0s) + j)
				K = [0] * len(p0s)
				K[j] = Ellipsis
				K = tuple(K)
				ax.errorbar(p0s[j], fp[(0, 0) + K + (i,)] - (np.asarray(p0s[i]) if i == j else p0s[i][0]), np.sqrt(cp[(0, 0) + K + (i, i)]), fmt=',')
				ax.set_xlabel('True $p_{%d}$' % j)
				ax.set_ylabel('$p_{%d}\'-p_{%d}$' % (i, i))
				pstr = ''
				for k in range(len(p0s)):
					if k != j:
						pstr += '$p_{%d}$ = %.2g\n' % (k, p0s[k][0])
				plot_text(pstr, ax=ax)
				ax.ticklabel_format(style='sci', axis='both', scilimits=(-3,3))
				ax.grid()
				
		plot_vsp0 = fig
	else:
		plot_vsp0 = None

	if plot.get('vsds', False):
		from matplotlib import pyplot as plt
	
		fig = plt.figure(figsize=(8, 3 * len(p0s)))
		fig.clf()
		fig.set_tight_layout(True)
		fig.canvas.set_window_title('%s, method “%s”, parameters vs. errors' % (fstr, method))
	
		ds = [
			[dxs, dys, 'x', 'y', (Ellipsis, 0)],
			[dys, dxs, 'y', 'x', (0, Ellipsis)]
		]
		for i in range(len(p0s)):
			for j in range(2):
				ax = fig.add_subplot(len(p0s), 2, 2*i + j + 1)
				if i == 0:
					pstr = ''
					for k in range(len(p0s)):
						pstr += '$p_{%d}$ = %.2g\n' % (k, p0s[k][0])
					pstr += '$\sqrt{\sum\Delta %s^2/n}=$%.2g' % (ds[j][3], np.sqrt((ds[j][1][0]**2).sum() / n))
					plot_text(pstr, ax=ax)
				if i == len(p0s) - 1:
					ax.set_xlabel('$\sqrt{\sum\Delta %s^2/n}$' % ds[j][2])
				if j == 0:
					ax.set_ylabel('$p_{%d}\'-p_{%d}$' % (i, i))
				sel = ds[j][4] + tuple([0] * len(p0s)) + (i,)
				Y = fp[sel] - np.asarray(p0s[i])
				DY = np.sqrt(cp[sel + (i,)])
				ax.errorbar(np.sqrt((ds[j][0]**2).sum(axis=-1) / n), Y, DY, fmt=',')
				pvalue = stats.chi2.sf(sum((Y / DY)**2), len(Y))
				plot_text('p-value = %.2g %%' % (pvalue * 100), loc=1, ax=ax)
		
		plot_vsds = fig
	else:
		plot_vsds = None
	
	out = FitCurveBootstrapOutput()
	out.fp = fp
	out.cp = cp
	plotout = dict()
	if len(plots_single) > 0:
		plotout['single'] = plots_single
	if not (plot_vsp0 is None):
		plotout['vsp0'] = plot_vsp0
	if not (plot_vsds is None):
		plotout['vsds'] = plot_vsds
	out.plot = plotout
	
	return out

def _fit_affine_odr(x, y, dx, dy):
	dy2 = dy**2
	dx2 = dx**2
	def residual(p):
		return (y - (p[0]*x + p[1])) / np.sqrt(dy2 + (p[0]*dx)**2)
	rt = np.empty((2, len(x)))
	def jac(p):
		rad = dy2 + p[0]**2 * dx2
		srad = np.sqrt(rad)
		res = (y - (p[0]*x + p[1])) * dx2 * p[0] / srad
		rt[0] = - (x * srad + res) / rad
		rt[1] = - 1 / srad
		return rt
	p0, _ = _fit_affine_unif_err(x, y)
	par, cov, _, _, _ = optimize.leastsq(residual, p0, Dfun=jac, col_deriv=True, full_output=True)
	return par, cov

def _fit_linear_odr(x, y, dx, dy):
	dy2 = dy**2
	dx2 = dx**2
	def residual(p):
		return (y - p[0]*x) / np.sqrt(dy2 + (p[0]*dx)**2)
	rt = np.empty((1, len(x)))
	def jac(p):
		rad = dy2 + p[0]**2 * dx2
		srad = np.sqrt(rad)
		res = (y - p[0]*x) * dx2 * p[0] / srad
		rt[0] = - (x * srad + res) / rad
		return rt
	p0, _ = _fit_affine_unif_err(x, y)
	par, cov, _, _, _ = optimize.leastsq(residual, (p0[0],), Dfun=jac, col_deriv=True, full_output=True)
	return np.array([par[0], 0]), np.array([[cov[0,0], 0], [0, 0]])

def _fit_affine_yerr(x, y, sigmay):
	dy2 = sigmay ** 2
	sy = (y / dy2).sum()
	sx2 = (x ** 2 / dy2).sum()
	sx = (x / dy2).sum()
	sxy = (x * y / dy2).sum()
	s1 = (1 / dy2).sum()
	denom = s1 * sx2 - sx ** 2
	a = (s1 * sxy - sy * sx) / denom
	b = (sy * sx2 - sx * sxy) / denom
	vaa = s1 / denom
	vbb = sx2 / denom
	vab = -sx / denom
	return np.array([a, b]), np.array([[vaa, vab], [vab, vbb]])

def _fit_linear_yerr(x, y, sigmay):
	dy2 = sigmay ** 2
	sx2 = (x ** 2 / dy2).sum()
	sxy = (x * y / dy2).sum()
	a = sxy / sx2
	vaa = 1 / sx2
	return np.array([a, 0]), np.array([[vaa, 0], [0, 0]])

def _fit_affine_unif_err(x, y):
	sy = y.sum()
	sx2 = (x ** 2).sum()
	sx = x.sum()
	sxy = (x * y).sum()
	s1 = len(x)
	denom = len(x) * sx2 - sx ** 2
	a = (s1 * sxy - sy * sx) / denom
	b = (sy * sx2 - sx * sxy) / denom
	vaa = s1 / denom
	vbb = sx2 / denom
	vab = -sx / denom
	return np.array([a, b]), np.array([[vaa, vab], [vab, vbb]])

def _fit_linear_unif_err(x, y):
	sx2 = (x ** 2).sum()
	sxy = (x * y).sum()
	a = sxy / sx2
	vaa = 1 / sx2
	return np.array([a, 0]), np.array([[vaa, 0], [0, 0]])

def _fit_affine_xerr(x, y, dx):
	par, cov = _fit_affine_yerr(y, x, dx)
	m, q = par
	dmm, dmq, _, dqq = cov.flat
	# par = np.array([1/m, -q/m])
	# J = np.array([[-1/m**2, 0], [q/m**2, -1/m]])
	# cov = J.dot(cov).dot(J.T)
	par[0] = 1 / m
	par[1] = -q / m
	m4 = m**4
	cov[0,0] = dmm / m4
	cov[1,1] = (-2 * dmq * m * q + dqq * m**2 + dmm * q**2) / m4
	cov[0,1] = (-dmm * q + dmq * m) / m4
	cov[1,0] = cov[0,1]
	return par, cov

def _fit_linear_xerr(x, y, dx):
	par, cov = _fit_linear_yerr(y, x, dx)
	m = par[0]
	dmm = cov[0,0]
	par[0] = 1 / m
	cov[0,0] = dmm / m**4
	return par, cov

def _fit_affine_ev(fun_fit, x, y, dx, dy, par, cov, absolute_sigma=True, conv_diff=1e-7, max_cycles=5):
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

_fit_lin_funcs = [
	[
		_fit_linear_odr,
		_fit_linear_yerr,
		_fit_linear_xerr,
		_fit_linear_unif_err
	],
	[
		_fit_affine_odr,
		_fit_affine_yerr,
		_fit_affine_xerr,
		_fit_affine_unif_err
	]
]

_fit_lin_ddofs = [1, 2]

def fit_linear(x, y, dx=None, dy=None, offset=True, absolute_sigma=True, method='odr', print_info=False, **kw):
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
		If both dx and dy are None, the fit behaves as if absolute_sigma=False,
		dx=0 and dy was uniform. If only one of dx or dy is None, the fit
		behaves as if it is zero.
	offset : bool
		If True, fit y = m * x + q; else fit y = m * x. If False,
		the output format does not change, and quantities corresponding
		to q are set to 0.
	absolute_sigma : bool
		If True, compute standard error on parameters (maximum likelihood
		estimation assuming datapoints are normal). If False, rescale
		errors on parameters to values that would be obtained if the
		chisquare matched the degrees of freedom.
		Simply said: True for physicists, False for engineers
	method : string, one of 'odr', 'ev'
		fit method to use when there are errors on both x and y.
		'odr': use orthogonal distance regression
		'ev': use effective variance
	print_info : bool
		If True, print information about the fit.

	Keyword arguments
	-----------------
	When method='ev', the following parameters are meaningful:
	conv_diff : number
		relative difference for convergence
	max_cycles : integer
		The maximum number of fits done. If this maximum is reached, an exception
		is raised.

	Returns
	-------
	par:
		estimates (m, q)
	cov:
		covariance matrix m, q
	"""
	x = np.asarray(x)
	y = np.asarray(y)
	offset = bool(offset)
	fitfun = _fit_lin_funcs[offset]
	ddof = _fit_lin_ddofs[offset]
	if not (dy is None):
		if not (dx is None): # dx, dy
			dx = np.asarray(dx)
			dy = np.asarray(dy)
			if method == 'odr':
				par, cov = fitfun[0](x, y, dx, dy)
				if not absolute_sigma:
					chisq_rid = ((y - par[0]*x - par[1])**2 / (dy**2 + (par[0]*dx)**2)).sum() / (len(x) - ddof)
					cov *= chisq_rid
			elif method == 'ev':
				ndy0 = np.asarray(dy != 0)
				if ndy0.sum() > ddof:
					par, cov = fitfun[1](x[ndy0], y[ndy0], dy[ndy0])
					if not absolute_sigma:
						chisq_rid = (((y - par[0]*x - par[1]) / dy)**2).sum() / (len(x) - ddof)
						cov *= chisq_rid
				else:
					par, cov = fitfun[3](x, y)
					chisq_rid = (((y - par[0]*x - par[1]))**2).sum() / (len(x) - ddof)
					cov *= chisq_rid
				par, cov, cycles = _fit_affine_ev(fitfun[1], x, y, dx, dy, par, cov, absolute_sigma=absolute_sigma, **kw)
				if cycles == -1:
					raise RuntimeError('Max cycles %d reached' % max_cycles)
				if print_info:
					print("fit_linear: cycles: %d" % (cycles))
			else:
				raise KeyError(method)
		else: # dy
			dy = np.asarray(dy)
			par, cov = fitfun[1](x, y, dy)
			if not absolute_sigma:
				chisq_rid = (((y - par[0]*x - par[1]) / dy)**2).sum() / (len(x) - ddof)
				cov *= chisq_rid
	else:
		if not (dx is None): # dx
			dx = np.asarray(dx)
			par, cov = fitfun[2](x, y, dx)
			if not absolute_sigma:
				chisq_rid = (((y - par[0]*x - par[1]) / (par[0]*dx))**2).sum() / (len(x) - ddof)
				cov *= chisq_rid
		else: # no error
			par, cov = fitfun[3](x, y)
			chisq_rid = ((y - par[0]*x - par[1])**2).sum() / (len(x) - ddof)
			cov *= chisq_rid
	return par, cov

def fit_const_yerr(y, sigmay):
	"""
	fit y = a

	Parameters
	----------
	y : M-length array
		dependent data
	sigmay : M-length array
		standard deviation of y

	Returns
	-------
	a : float
		optimal value for a
	vara : float
		variance of a
	"""
	y = np.asarray(y)
	sigmay = np.asarray(sigmay)
	dy2 = sigmay ** 2
	sy = (y / dy2).sum()
	s1 = (1 / dy2).sum()
	a = sy / s1
	vara = 1 / s1
	return a, vara

def fit_oversampling(data, digit=1, print_info=0, plot_axes=None):
	"""
	Given discretized samples, find the maximum likelihood estimate
	of the average and standard deviation of a normal distribution.
	
	Parameters
	----------
	data : 1D array-like
		Discretized samples. The discretization is assumed to be a
		rounding to nearest integer, or to fixed multiple of integer (see
		digit parameter).
	digit : number
		The unit of discretization. Data is divided by digit before
		computing, then results are multiplied by digit.
	print_info : integer
		
	plot_axes : Axes3DSubplot or None
		If a 3D subplot is given, plot the likelihood around the estimate.
	
	Returns
	-------
	par : 1D array
		Estimated [mean, standard deviation].
	cov : 2D array
		Covariance matrix of the estimate.
	"""
	import numdifftools as numdiff
	
	if print_info >= 1:
		print('############### fit_oversampling ###############')
		print()
	
	data = np.asarray(data) / digit
	n = len(data)
	
	p0 = [data.mean(), data.std(ddof=1)]
	if p0[1] == 0:
		p0[1] = np.sqrt(n - 1) / n / 2
	
	data -= p0[0]
	data /= p0[1]
	hdigit = 1 / p0[1] / 2
	
	points, counts = np.unique(data, return_counts=True)
	
	if print_info >= 1:
		print('Number of data points: %d' % n)
		print('Number of unique points: %d' % len(points))
		print('Discretization unit: %.3g' % digit)
		print('Sample mean: %.3g' % (p0[0] * digit))
		print('Sample standard deviation: %.3g' % (p0[1] * digit if len(counts) > 1 else 0))
		print()
	
	if print_info == 1:
		print('Minimizing...')
	
	lp = np.empty(len(points))
	
	def minusloglikelihood(par):
		mu, sigma = par
		dist = stats.norm(loc=mu, scale=sigma)
		for i in range(len(lp)):
			x = points[i]
			if x > mu:
				p = dist.sf(x - hdigit) - dist.sf(x + hdigit)
			else:
				p = dist.cdf(x + hdigit) - dist.cdf(x - hdigit)
			if p == 0:
				xd = np.abs(x - mu) - hdigit * 0.8
				p = dist.logpdf(0.5 * (np.sign(xd) + 1) * xd * np.sign(x - mu))
			else:
				p = math.log(p)
			lp[i] = p
				
		return -np.sum(counts * lp)# + np.log(sigma) # jeffreys' prior
		
	result = optimize.minimize(minusloglikelihood, (0, 1), method='L-BFGS-B', options=dict(disp=print_info >= 2), bounds=((-hdigit, hdigit), (0.1, None)))
	
	if print_info >= 2:
		print('###### MINIMIZATION FINISHED ######')
		print()
	
	par = result.x
	hess = numdiff.Hessian(minusloglikelihood, method='forward')(par)
	try:
		cov = linalg.inv(hess)
	except linalg.LinAlgError:
		if print_info >= 1:
			print('Hessian is not invertible, computing pseudo-inverse')
			print()
		W, V = linalg.eigh(hess)
		cov = np.zeros(hess.shape)
		np.fill_diagonal(cov, [(1 / w if w != 0 else 0) for w in W])
		cov = V.dot(cov).dot(V.T)
	
	if not (plot_axes is None):
		if print_info >= 1:
			print('Plotting likelihood...')
			print()
		plot_axes.cla()
		factor = special.gammaln(1 + n) - np.sum(special.gammaln(1 + counts))
		err = np.sqrt(abs(np.diag(cov)))
		if len(counts) > 1:
			x = np.linspace(max(-hdigit, par[0] - 2 * err[0]), min(hdigit, par[0] + 2 * err[0]), 41)
			y = np.linspace(max(0.05, par[1] - 2 * err[1]), min(12, par[1] + 2 * err[1]), 39)
		else:
			x = np.linspace(-hdigit, hdigit, 41)
			y = np.linspace(0.05, 12, 39)
		X, Y = np.meshgrid(x, y)
		plot_axes.plot_surface(X, Y, [[np.exp(factor - minusloglikelihood((X[i, j], Y[i, j]))) for j in range(len(X[0]))] for i in range(len(X))])
		plot_axes.plot3D([par[0]], [par[1]], [np.exp(factor - minusloglikelihood(par))], 'ok')
		plot_axes.set_xlabel('Mean')
		plot_axes.set_ylabel('Sigma')
		plot_axes.set_zlabel('Likelihood')

	par *= p0[1]
	par[0] += p0[0]
	par *= digit
	
	cov *= p0[1] ** 2
	cov *= digit ** 2
	
	if print_info >= 2:
		print('###### SUMMARY ######')
		print('Sample mean: %.3g' % (p0[0] * digit))
		print('Sample standard deviation: %.3g' % (p0[1] * digit if len(counts) > 1 else 0))
	if print_info >= 1:
		print('Estimated mean, standard deviation (with correlation):')
		print(format_par_cov(par, cov))
		
	if print_info >= 1:
		print()
		print('############# END fit_oversampling #############')
		
	return par, cov

# *********************** MULTIMETERS *************************

def _find_scale(x, scales):
	# (!) scales sorted ascending
	for i in range(len(scales)):
		if x < scales[i]:
			return i
	return -1

def _find_scale_idx(scale, scales):
	# (!) scales sorted ascending
	for i in range(len(scales)):
		if scale == scales[i]:
			return i
		elif scale < scales[i]:
			return -1
	return -1

_util_mm_esr_data = dict(
	dm3900=dict(
		desc='multimeter Digimaster DM 3900 plus',
		type='digital',
		voltres=10e+6,
		volt=dict(
			scales=[0.2, 2, 20, 200, 1000],
			perc=[0.5] * 4 + [0.8],
			digit=[1, 1, 1, 1, 2]
		),
		volt_ac=dict(
			scales=[0.2, 2, 20, 200, 700],
			perc=[1.2, 0.8, 0.8, 0.8, 1.2],
			digit=[3] * 5
		),
		cdt=0.2,
		ampere=dict(
			scales=[2 * 10**z for z in range(-5, 2)],
			perc=[2, 0.5, 0.5, 0.5, 1.2, 1.2, 2],
			digit=[5, 1, 1, 1, 1, 1, 5]
		),
		ampere_ac=dict(
			scales=[2 * 10**z for z in range(-5, 2)],
			perc=[3, 1.8, 1, 1, 1.8, 1.8, 3],
			digit=[7, 3, 3, 3, 3, 3, 7]
		),
		ohm=dict(
			scales=[2 * 10**z for z in range(2, 8)],
			perc=[0.8] * 5 + [1],
			digit=[3, 1, 1, 1, 1, 2]
		)
	),
	lab3=dict(
		desc='multimeter from lab III course',
		type='digital',
		voltres=10e+6,
		volt=dict(
			scales=[0.2, 2, 20, 200, 1000],
			perc=[0.5] * 4 + [0.8],
			digit=[1, 1, 1, 1, 2]
		),
		volt_ac=dict(
			scales=[0.2, 2, 20, 200, 700],
			perc=[1.2, 0.8, 0.8, 0.8, 1.2],
			digit=[3] * 5
		),
		cdt=0.2,
		ampere=dict(
			scales=[2e-3, 20e-3, 0.2, 10],
			perc=[0.8, 0.8, 1.5, 2.0],
			digit=[1, 1, 1, 5]
		),
		ampere_ac=dict(
			scales=[2e-3, 20e-3, 0.2, 10],
			perc=[1, 1, 1.8, 3],
			digit=[3, 3, 3, 7]
		),
		ohm=dict(
			scales=[2 * 10**z for z in range(2, 9)],
			perc=[0.8] * 5 + [1, 5],
			digit=[3, 1, 1, 1, 1, 2, 10]
		),
		farad=dict(
			scales=[2e-9 * 10**z for z in range(1, 6)],
			perc=[4] * 5,
			digit=[3] * 5
		)
	),
	kdm700=dict(
		desc='multimeter GBC Mod. KDM-700NCV',
		type='digital',
		voltres=10e+6,
		volt=dict(
			scales=[0.2, 2, 20, 200, 1000],
			perc=[0.5] * 4 + [0.8],
			digit=[1, 1, 1, 1, 2]
		),
		volt_ac=dict(
			scales=[0.2, 2, 20, 200, 700],
			perc=[1.2, 0.8, 0.8, 0.8, 1.2],
			digit=[3] * 5
		),
		cdt=0.2,
		ampere=dict(
			scales=[2 * 10**z for z in range(-5, 0)] + [10],
			perc=[2, 0.8, 0.8, 0.8, 1.5, 2],
			digit=[5, 1, 1, 1, 1, 5]
		),
		ampere_ac=dict(
			scales=[2 * 10**z for z in range(-5, 0)] + [10],
			perc=[2, 1, 1, 1, 1.8, 3],
			digit=[5] * 5 + [7]
		),
		ohm=dict(
			scales=[2 * 10**z for z in range(2, 9)],
			perc=[0.8] * 5 + [1, 5],
			digit=[3, 1, 1, 1, 1, 2, 10]
		)
	),
	ice680=dict(
		desc='multimeter ICE SuperTester 680R VII serie',
		type='analog',
		volt=dict(
			scales=[0.1, 2, 10, 50, 200, 500, 1000],
			relres=[50] * 7, # scale / resolution
			valg=[1] * 7 # guaranteed error / scale * 100
		),
		volt_ac=dict(
			scales=[10, 50, 250, 750],
			relres=[50] * 3 + [37.5],
			valg=[2] * 3 + [100.0 / 37.5]
		),
		ampere=dict(
			scales=[50e-6, 500e-6, 5e-3, 50e-3, 500e-3, 5],
			relres=[50] * 6,
			valg=[1] * 6,
			cdt=[0.1, 0.294, 0.318] + [0.320] * 3
		),
		ampere_ac=dict(
			scales=[250e-6, 2.5e-3, 25e-3, 250e-3, 2.5],
			relres=[50] * 5,
			valg=[2] * 5,
			cdt=[2, 1.5, 1.6, 1.6, 1.9]
		)
	),
	oscil=dict(
		desc='oscilloscope from lab III course',
		type='oscil',
		volt=dict(
			scales=[ (8*d*10**s) for s in range(-3, 1) for d in [1, 2, 5] ],
			perc=[4] * 2 + [3] * 10,
			div=[ (d*10**s) for s in range(-3, 1) for d in [1, 2, 5] ]
		),
		time=dict(
			scales=[5e-09] + [ (10*d*10**s) for s in range(-9, 2) for d in [1, 2.5, 5] ],
			div=[5e-10] + [ (d*10**s) for s in range(-9, 2) for d in [1, 2.5, 5] ]
		),
		freq=dict(
			scales=[1e9]
		),
		generic=dict(
		)
	)
)

def util_mm_list():
	"""
	Returns
	-------
	l : list
		List of tuples, one for each metertype understood by util_mm_er, containing:
		(metertype, type, description)
	"""
	l = []
	for meter in _util_mm_esr_data:
		l += [(meter, _util_mm_esr_data[meter]['type'], _util_mm_esr_data[meter]['desc'])]
	return l

def util_mm_er(x, scale, metertype='lab3', unit='volt', sqerr=False):
	"""
	Returns the uncertainty of x and the internal resistance of the multimeter.

	Parameters
	----------
	x : number
		the value measured, may be negative
	scale : number
		the fullscale used to measure x
	metertype : string
		one of the names returned by lab.util_mm_list()
		the multimeter used
	unit : string
		one of 'volt', 'volt_ac', 'ampere' 'ampere_ac', 'ohm', 'farad'
		the unit of measure of x
	sqerr : bool or callable
		If False, sum errors. If True, sum squares of errors. If callable,
		two errors are summed calling sqerr(err1, err2). For digital multimeters,
		the first argument passed to sqerr is the percentual error.

	Returns
	-------
	e : number
		the uncertainty
	r : number or None
		the internal resistance (if applicable)

	See also
	--------
	util_mm_esr, util_mm_esr2, mme
	"""

	x = abs(x)

	errsum = sqerr if hasattr(sqerr, '__call__') else (lambda x, y: math.sqrt(x**2 + y**2)) if sqerr else (lambda x, y: x + y)

	meter = _util_mm_esr_data[metertype]
	info = meter[unit]
	typ = meter['type']

	s = scale
	idx = _find_scale_idx(s, info['scales'])
	if idx < 0:
		raise KeyError(s)
	r = None

	if typ == 'digital':
		e = errsum(x * info['perc'][idx] / 100.0, info['digit'][idx] * 10**(idx + math.log10(info['scales'][0] / 2.0) - 3))
		if unit == 'volt' or unit == 'volt_ac':
			r = meter['voltres']
		elif unit == 'ampere' or unit == 'ampere_ac':
			r = info['cdt'] / s
	elif typ == 'analog':
		e = s * errsum(info['valg'][idx] / 100.0, 0.5 / info['relres'][idx])
		if unit == 'volt' or unit == 'volt_ac':
			r = 20000 * s
		elif unit == 'ampere' or unit == 'ampere_ac':
			r = info['cdt'][idx] / s
	elif typ == 'oscil':
		e = info['div'][idx] / 25
		r = 10e6
	else:
		raise KeyError(typ)

	return e, r

def util_mm_esr(x, metertype='lab3', unit='volt', sqerr=False):
	"""
	Determines the fullscale used to measure x with a multimeter,
	supposing the lowest possible fullscale was used, and returns the
	uncertainty, the fullscale and the internal resistance.

	Parameters
	----------
	x : number
		the value measured, may be negative
	metertype : string
		one of the names returned by util_mm_list()
		the multimeter used
	unit : string
		one of 'volt', 'volt_ac', 'ampere' 'ampere_ac', 'ohm', 'farad'
		the unit of measure of x
	sqerr : bool
		If True, sum errors squaring.

	Returns
	-------
	e : number
		the uncertainty
	s : number
		the full-scale
	r : number or None
		the internal resistance (if applicable)

	See also
	--------
	util_mm_er, util_mm_esr2, mme
	"""

	x = abs(x)
	info = _util_mm_esr_data[metertype][unit]
	idx = _find_scale(x, info['scales'])
	if idx < 0:
		raise ValueError("value '%.4g %s' too big for all scales" % (x, unit))
	s = info['scales'][idx]
	e, r = util_mm_er(x, s, metertype=metertype, unit=unit, sqerr=sqerr)
	return e, s, r

_util_mm_esr_vect_error = np.vectorize(lambda x, y, z, t: util_mm_esr(x, metertype=y, unit=z, sqerr=t)[0], otypes=[np.number])
_util_mm_esr_vect_scale = np.vectorize(lambda x, y, z, t: util_mm_esr(x, metertype=y, unit=z, sqerr=t)[1], otypes=[np.number])
_util_mm_esr_vect_res = np.vectorize(lambda x, y, z, t: util_mm_esr(x, metertype=y, unit=z, sqerr=t)[2], otypes=[np.number])
_util_mm_esr2_what = dict(
	error=_util_mm_esr_vect_error,
	scale=_util_mm_esr_vect_scale,
	res=_util_mm_esr_vect_res
)

def util_mm_esr2(x, metertype='lab3', unit='volt', what='error', sqerr=False):
	"""
	Vectorized version of lab.util_mm_esr

	Parameters
	----------
	what : string
		one of 'error', 'scale', 'res'
		what to return

	Returns
	-------
	z : number
		either the uncertainty, the fullscale or the internal resistance.

	See also
	--------
	util_mm_er, util_mm_esr, mme
	"""
	if unit == 'ohm' and what == 'res':
		raise ValueError('asking internal resistance of ohmmeter')
	return _util_mm_esr2_what[what](x, metertype, unit, sqerr)

def mme(x, unit, metertype='lab3', sqerr=False):
	"""
	determines the fullscale used to measure x with a multimeter,
	supposing the lowest possible fullscale was used, and returns the
	uncertainty of the measurement.

	Parameters
	----------
	x : (X-shaped array of) number
		the value measured, may be negative
	unit : (X-shaped array of) string
		one of 'volt', 'volt_ac', 'ampere' 'ampere_ac', 'ohm', 'farad'
		the unit of measure of x
	metertype : (X-shaped array of) string
		one of the names returned by util_mm_list()
		the multimeter used
	sqerr : bool
		If True, sum errors squaring.

	Returns
	-------
	e : (X-shaped array of) number
		the uncertainty

	See also
	--------
	util_mm_er, util_mm_esr, util_mm_esr2
	"""
	return util_mm_esr2(x, metertype=metertype, unit=unit, what='error', sqerr=sqerr)

# *********************** FORMATTING *************************

d = lambda x, n: int(("%.*e" % (n - 1, abs(x)))[0])
ap = lambda x, n: float("%.*e" % (n - 1, x))
_nd = lambda x: math.floor(math.log10(abs(x))) + 1
def _format_epositive(x, e, errsep=True, minexp=3):
	# DECIDE NUMBER OF DIGITS
	if d(e, 2) < 3:
		n = 2
		e = ap(e, 2)
	elif d(e, 1) < 3:
		n = 2
		e = ap(e, 1)
	else:
		n = 1
	# FORMAT MANTISSAS
	dn = int(_nd(x) - _nd(e)) if x != 0 else -n
	nx = n + dn
	if nx > 0:
		ex = _nd(x) - 1
		if nx > ex and abs(ex) <= minexp:
			xd = nx - ex - 1
			ex = 0
		else:
			xd = nx - 1
		sx = "%.*f" % (xd, x / 10**ex)
		se = "%.*f" % (xd, e / 10**ex)
	else:
		le = _nd(e)
		ex = le - n
		sx = '0'
		se = "%#.*g" % (n, e)
	# RETURN
	if errsep:
		return sx, se, ex
	return sx + '(' + ("%#.*g" % (n, e * 10 ** (n - _nd(e))))[:n] + ')', '', ex

def util_format(x, e, pm=None, percent=False, comexp=True, nicexp=False):
	"""
	format a value with its uncertainty

	Parameters
	----------
	x : number (or something understood by float(), ex. string representing number)
		the value
	e : number (or as above)
		the uncertainty
	pm : string, optional
		The "plusminus" symbol. If None, use compact notation.
	percent : bool
		if True, also format the relative error as percentage
	comexp : bool
		if True, write the exponent once.
	nicexp : bool
		if True, format exponent like ×10¹²³

	Returns
	-------
	s : string
		the formatted value with uncertainty

	Examples
	--------
	util_format(123, 4) --> '123(4)'
	util_format(10, .99) --> '10.0(10)'
	util_format(1e8, 2.5e6) --> '1.000(25)e+8'
	util_format(1e8, 2.5e6, pm='+-') --> '(1.000 +- 0.025)e+8'
	util_format(1e8, 2.5e6, pm='+-', comexp=False) --> '1.000e+8 +- 0.025e+8'
	util_format(1e8, 2.5e6, percent=True) --> '1.000(25)e+8 (2.5 %)'
	util_format(nan, nan) --> 'nan +- nan'

	See also
	--------
	xe, xep
	"""
	x = float(x)
	e = abs(float(e))
	if not math.isfinite(x) or not math.isfinite(e) or e == 0:
		return "%.3g %s %.3g" % (x, '+-', e)
	sx, se, ex = _format_epositive(x, e, not (pm is None))
	if ex == 0:
		es = ''
	elif nicexp:
		es = "×10" + num2sup(ex, format='%d')
	else:
		es = "e%+d" % ex
	if pm is None:
		s = sx + es
	elif comexp and es != '':
		s = '(' + sx + ' ' + pm + ' ' + se + ')' + es
	else:
		s = sx + es + ' ' + pm + ' ' + se + es
	if (not percent) or sx.split('(')[0] == '0':
		return s
	pe = e / abs(x) * 100.0
	return s + " (%.*g %%)" % (2 if pe < 100.0 else 3, pe)

_util_format_vect = np.vectorize(util_format, otypes=[str])

def xe(x, e, pm=None, comexp=True, nicexp=False):
	"""
	Vectorized version of util_format with percent=False,
	see lab.util_format and numpy.vectorize.

	Example
	-------
	xe(['1e7', 2e7], 33e4) --> ['1.00(3)e+7', '2.00(3)e+7']
	xe(10, 0.8, pm=unicode_pm) --> '10.0 ± 0.8'

	See also
	--------
	xep, num2si, util_format
	"""
	return _util_format_vect(x, e, pm, False, comexp, nicexp)

def xep(x, e, pm=None, comexp=True, nicexp=False):
	"""
	Vectorized version of util_format with percent=True,
	see lab.util_format and numpy.vectorize.

	Example
	-------
	xep(['1e7', 2e7], 33e4) --> ['1.00(3)e+7 (3.3 %)', '2.00(3)e+7 (1.7 %)']
	xep(10, 0.8, pm=unicode_pm) --> '10.0 ± 0.8 (8 %)'

	See also
	--------
	xe, num2si, util_format
	"""
	return _util_format_vect(x, e, pm, True, comexp, nicexp)

unicode_pm = u'±'

# this function taken from stackoverflow and modified
# http://stackoverflow.com/questions/17973278/python-decimal-engineering-notation-for-mili-10e-3-and-micro-10e-6
def num2si(x, format='%.15g', si=True, space=' '):
	"""
	Returns x formatted using an exponent that is a multiple of 3.

	Parameters
	----------
	x : number
		the number to format
	format : string
		printf-style string used to format the mantissa
	si : boolean
		if true, use SI suffix for exponent, e.g. k instead of e3, n instead of
		e-9 etc. If the exponent would be greater than 24, numerical exponent is
		used anyway.
	space : string
		string interposed between the mantissa and the exponent

	Returns
	-------
	fx : string
		the formatted value

	Example
	-------
	     x     | num2si(x)
	-----------|----------
	   1.23e-8 |  12.3 n
	       123 |  123
	    1230.0 |  1.23 k
	-1230000.0 |  -1.23 M
	         0 |  0

	See also
	--------
	util_format, xe, xep
	"""
	x = float(x)
	if x == 0:
		return format % x + space
	exp = int(math.floor(math.log10(abs(x))))
	exp3 = exp - (exp % 3)
	x3 = x / (10 ** exp3)

	if si and exp3 >= -24 and exp3 <= 24 and exp3 != 0:
		exp3_text = space + 'yzafpnμm kMGTPEZY'[(exp3 - (-24)) // 3]
	elif exp3 == 0:
		exp3_text = space
	else:
		exp3_text = 'e%s' % exp3 + space

	return (format + '%s') % (x3, exp3_text)

_subscr  = '₀₁₂₃₄₅₆₇₈₉₊₋ₑ․'
_subscrc = '0123456789+-e.'
_supscr  = '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ᵉ·'

def num2sub(x, format=None):
	"""
	Format a number as subscript.

	Parameters
	----------
	x : string or number
		The number to format.
	format : None or string
		If None, x is interpreted as string and formatted subscript as-is.
		If string, it is a %-format used to format x before converting to subscript.

	Returns
	-------
	s : string
		x written in subscript.
	"""
	if format is None:
		x = str(x)
	else:
		x = format % float(x)
	for i in range(len(_subscrc)):
		x = x.replace(_subscrc[i], _subscr[i])
	return x

def num2sup(x, format=None):
	"""
	Format a number as superscript.

	Parameters
	----------
	x : string or number
		The number to format.
	format : None or string
		If None, x is interpreted as string and formatted superscript as-is.
		If string, it is a %-format used to format x before converting to superscript.

	Returns
	-------
	s : string
		x written in superscript.
	"""
	if format is None:
		x = str(x)
	else:
		x = format % float(x)
	for i in range(len(_subscrc)):
		x = x.replace(_subscrc[i], _supscr[i])
	return x

def format_par_cov(par, cov):
	"""
	Format an estimate with a covariance matrix as an upper
	triangular matrix with parameters on the diagonal (with
	uncertainties) and correlations off-diagonal.
	
	Parameters
	----------
	par : M-length array
		Parameters to be written on the diagonal.
	cov : (M, M) matrix
		Covariance from which uncertainties and correlations
		are computed.
	
	Examples
	--------
	>>> out = fit_curve(f, x, y, ...)
	>>> print(format_par_cov(out.par, out.cov))
	"""
	pars = xe(par, np.sqrt(np.diag(cov)))
	corr = fit_norm_cov(cov) * 100
	corrwidth = 8
	s = ''
	for i in range(len(par)):
		for j in range(len(par)):
			width = max(corrwidth, len(pars[j])) + 1
			if i == j:
				sadd = pars[i]
			elif i < j:
				sadd = "%*.1f %%" % (corrwidth - 2, corr[i, j])
			else:
				sadd = ''
			s += ' ' * (width - len(sadd)) + sadd
			if j != len(par) - 1:
				s += ' '
			elif i != len(par) - 1:
				s += '\n'
	return s

# ************************** TIME *********************************

def util_timecomp(secs):
	"""
		convert a time interval in seconds to hours, minutes, seconds

		Parameters
		----------
		secs : number
			the time interval expressed in seconds

		Returns
		-------
		hours : int
			hours, NOT bounded to 24
		minutes : int
			minutes, 0--59
		seconds : int
			seconds, 0--59

		See also
		--------
		util_timestr
	"""
	hours = int(secs / 3600)
	minutes = int((secs - hours * 3600) / 60)
	seconds = secs - hours * 3600 - minutes * 60
	return hours, minutes, seconds

def util_timestr(secs):
	"""
		convert a time interval in seconds to a string with hours, minutes, seconds

		Parameters
		----------
		secs : number
			the time interval expressed in seconds

		Returns
		-------
		str : str
			string representing the interval

		See also
		--------
		util_timecomp
	"""
	return "%02d:%02d:%02d" % util_timecomp(secs)

class Eta():
	
	def __init__(self):
		"""
		Object to compute the eta (estimated time of arrival).
		Create the object at the start of a lengthy process,
		then use one of the methods to compute the estimated
		remaining time.
		
		Examples
		--------
		>>> eta = lab.Eta() # initialize just before the start
		>>> for i in range(N):
		>>>     progress = i / N # a number between 0 and 1
		>>>     eta.etaprint(progress, mininterval=10.0) # print remaining time every 10 seconds
		>>>     # lengthy task(i)
		"""
		self.restart()
	
	def restart(self):
		"""
		Reset the eta object as if it was just initialized.
		"""
		now = time.time()
		self._start_time = now
		self._last_stamp = now
	
	def eta(self, progress):
		"""
		Compute the estimated time of arrival.

		Parameters
		----------
		progress : number in [0,1]
			The progress on a time-linear scale where 0 means still nothing done and
			1 means finished.

		Returns
		-------
		eta : float
			The time remaining to the arrival, in seconds.
		"""
		if 0 < progress <= 1:
			now = time.time()
			interval = now - self._start_time
			return (1 - progress) * interval / progress
		elif progress == 0:
			return np.inf
		else:
			raise RuntimeError("progress %.2f out of bounds [0,1]" % progress)
	
	def etastr(self, progress):
		"""
		Compute the estimated time of arrival.

		Parameters
		----------
		progress : number in [0,1]
			The progress on a time-linear scale where 0 means still nothing done and
			1 means finished.

		Returns
		-------
		etastr : string
			The time remaining to the arrival, formatted.
		"""
		eta = self.eta(progress)
		if np.isfinite(eta):
			return util_timestr(eta)
		else:
			return "--:--:--"
	
	def etaprint(self, progress, mininterval=5.0):
		"""
		Print the time elapsed and the estimated time of arrival.

		Parameters
		----------
		progress : number in [0,1]
			The progress on a time-linear scale where 0 means still nothing done and
			1 means finished.
		mininterval : number
			If the elapsed time since the last time etaprint did print a message is
			less than mininterval, do not print the message. Give a negative value
			to print in any case.
		"""
		now = time.time()
		etastr = self.etastr(progress)
		if now - self._last_stamp >= mininterval:
			print('elapsed time: %s, remaining time: %s' % (util_timestr(now - self._start_time), etastr))
			self._last_stamp = now

# *************************** FILES *******************************

def sanitizefilename(name, windows=True):
	"""
	Removes characters not allowed by the filesystem, replacing
	them with similar unicode characters.

	Parameters
	----------
	name : string
		The file name to sanitize. It can not be a path, since slashes are
		replaced.
	windows : bool
		If True, also replace characters not allowed in Windows.

	Return
	------
	filename : string
		The sanitized file name.
	"""
	name = name.replace('/', '∕').replace('\0', '').replace(':', '﹕')
	if windows:
		name = name.replace('\\', '⧵')
	return name

def nextfilename(base, ext, idxfmt='%02d', prepath=None, start=1, sanitize=True):
	"""
	Consider the following format:
		<base><index><ext>
	This functions search for the pattern with the lowest index that is not
	the path of an existing file.

	Parameters
	----------
	base : string
		Tipically the name of the file, without extension.
	ext : string
		Tipically the file type extension (with dot).
	idxfmt : string
		The %-format used to format the index.
	prepath : None or string
		A path that is prepended to <base> with a slash:
			<prepath>/<base>...
	start : number
		The index to start with.
	sanitize : bool
		If True, process <base> and <ext> with sanitizefilename. In this case,
		<base> should not contain a path since slashes are replaced. Use
		<prepath> instead.

	Returns
	-------
	filename : string
		File name of non-existing file.
	"""
	if sanitize:
		base = sanitizefilename(base)
		ext = sanitizefilename(ext)
	i = start
	while True:
		filename = ('%s%s-' + idxfmt + '%s') % ((prepath + '/') if prepath != None else '', base, i, ext)
		if not os.path.exists(filename):
			break
		i += 1
	return filename

# ************************ COMPATIBILITY ****************************

def fit_generic_xyerr(f, dfdx, x, y, sigmax, sigmay, p0=None, print_info=False, absolute_sigma=True, conv_diff=0.001, max_cycles=5, **kw):
	"""
	THIS FUNCTION IS DEPRECATED
	"""
	model = CurveModel(f, dfdx=dfdx, symb=False)
	return fit_curve(model, x, y, dx=sigmax, dy=sigmay, p0=p0, absolute_sigma=absolute_sigma, print_info=print_info, method='ev', conv_diff=conv_diff, max_cycles=max_cycles, **kw)

def fit_generic_xyerr2(f, x, y, sigmax, sigmay, p0=None, print_info=False, absolute_sigma=True):
	"""
	THIS FUNCTION IS DEPRECATED
	"""
	model = CurveModel(f, symb=False)
	return fit_curve(model, x, y, dx=sigmax, dy=sigmay, p0=p0, absolute_sigma=absolute_sigma, print_info=print_info, method='odrpack')

curve_fit_patched = optimize.curve_fit

def etastart():
	"""
	THIS FUNCTION IS DEPRECATED
	"""
	return Eta()

def etastr(eta, progress, mininterval=np.inf):
	"""
	THIS FUNCTION IS DEPRECATED
	"""
	eta.etaprint(progress, mininterval=mininterval)
	return util_timestr(time.time() - eta._start_time), eta.etastr(progress)
