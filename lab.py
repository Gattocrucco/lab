# ********************** IMPORTS ***************************

import math
import inspect
import numpy as np
import time
from scipy import odr
from scipy.optimize import curve_fit

__all__ = [ # things imported when you do "from lab import *"
	'curve_fit_patched',
	'fit_norm_cov',
	'fit_generic_xyerr',
	'fit_generic_xyerr2',
	'fit_linear',
	'fit_const_yerr',
	'util_mm_er',
	'etastart',
	'etastr',
	'num2si',
	'mme',
	'unicode_pm',
	'xe',
	'xep',
	'util_format'
]

__version__ = '2016.11'

# ************************** FIT ***************************

def _check_finite(array): # asarray_chkfinite is absent in old numpies
	for x in array.flat:
		if not np.isfinite(x):
			raise ValueError("array must not contain infs or NaNs")

def curve_fit_patched(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False, check_finite=True, **kw):
	"""
		Same as curve_fit, but add absolute_sigma and check_finite if scipy is old.
		If the keyword argument force_patch=True is given, the patch is used anyway.
	"""
	force_patch = kw.pop('force_patch', False)
	args = inspect.getargspec(curve_fit).args
	if 'absolute_sigma' in args and 'check_finite' in args and not force_patch:
		rt = curve_fit(f, xdata, ydata, p0, sigma, absolute_sigma, check_finite, **kw)
	elif 'absolute_sigma' in args and not force_patch:
		if check_finite:
			_check_finite(xdata)
			_check_finite(ydata)
		rt = curve_fit(f, xdata, ydata, p0, sigma, absolute_sigma, **kw)
	else: # the case check_finite yes and absolute_sigma no does not exist
		myp0 = p0
		if p0 is None: # we need p0 to implement absolute_sigma
			args = inspect.getargspec(f).args
			if len(args) < 2:
				raise ValueError("Unable to determine number of fit parameters.")
			myp0 = [1.0] * (len(args) - (2 if 'self' in args else 1))
		if np.isscalar(myp0):
			myp0 = np.array([myp0])
		if check_finite:
			_check_finite(xdata)
			_check_finite(ydata)
		rt = curve_fit(f, xdata, ydata, p0, sigma, **kw)
		if absolute_sigma and len(ydata) > len(myp0): # invert the normalization done by curve_fit
			popt = rt[0]
			s_sq = sum(((np.asarray(ydata) - f(xdata, *popt)) / (np.asarray(sigma) if sigma != None else 1.0)) ** 2) / (len(ydata) - len(myp0))
			pcov = rt[1] / s_sq
			rt = np.concatenate(([popt, pcov], rt[2:]))
	return rt

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
	sigma = np.sqrt(np.diag(cov))
	return cov / np.outer(sigma, sigma)

def fit_generic_xyerr(f, dfdx, x, y, sigmax, sigmay, p0=None, print_info=False, absolute_sigma=True, conv_diff=0.001, max_cycles=5, **kw):
	"""
	fit y = f(x, *params)

	Parameters
	----------
	f : callable
		the function to fit
	dfdx : callable
		derivative of f respect to x: dfdx(x, *params)
	x : M-length array-like
		independent data
	y : M-length array-like
		dependent data
	sigmax : M-length array-like or None
		standard deviation of x
	sigmay : M-length array-like or None
		standard deviation of y
	p0 : N-length sequence
		initial guess for parameters
	print_info : bool
		If True, print information about the fit
	absolute_sigma : bool
		If False, compute asymptotic errors, else standard errors for parameters
	conv_diff : number
		the difference in terms of standard deviation that
		is considered sufficient for convergence; see notes
	max_cycles : integer
		the maximum number of fits done; see notes.
		If this maximum is reached, an exception is raised.

	Keyword arguments are passed directly to curve_fit (see notes).

	Returns
	-------
	par : N-length array
		optimal values for parameters
	cov : (N,N)-shaped array
		covariance matrix of par

	Notes
	-----
	Algorithm: run curve_fit once ignoring sigmax, then propagate sigmax using
	dfdx and run curve_fit again with:
		sigmay = sqrt(sigmay**2 + (propagated sigmax)**2)
	until the differences between two successive estimates of the parameters are
	less than conv_diff times the corresponding estimated errors.
	"""
	if sigmax is None:
		return curve_fit_patched(f, x, y, p0=p0, sigma=sigmay, absolute_sigma=absolute_sigma, **kw)
	x = np.asarray(x)
	sigmax = np.asarray(sigmax)
	if not (sigmay is None):
		sigmay = np.asarray(sigmay)
	cycles = 1
	rt = curve_fit_patched(f, x, y, p0=p0, sigma=sigmay, absolute_sigma=absolute_sigma, **kw)
	par, cov = rt[:2]
	sigma = np.sqrt(np.diag(cov))
	error = sigma # to pass loop condition
	p0 = par
	while any(error > sigma * conv_diff):
		if cycles >= max_cycles:
			raise RuntimeError("Maximum number of fit cycles %d reached" % max_cycles)
		psigmax = dfdx(x, *p0) * sigmax
		sigmayeff = psigmax if sigmay is None else np.sqrt(psigmax**2 + sigmay**2)
		rt = curve_fit_patched(f, x, y, p0=p0, sigma=sigmayeff, absolute_sigma=absolute_sigma, **kw)
		par, cov = rt[:2]
		sigma = np.sqrt(np.diag(cov))
		error = abs(par - p0)
		p0 = par
		cycles += 1
	if print_info:
		print(fit_generic_xyerr, ": cycles: %d" % (cycles))
	return rt

def fit_generic_xyerr2(f, x, y, sigmax, sigmay, p0=None, print_info=False, absolute_sigma=True):
	"""
		fit y = f(x, *params)

		Parameters
		----------
		f : callable
			the function to fit
		x : M-length array
			independent data
		y : M-length array
			dependent data
		sigmax : M-length array
			standard deviation of x
		sigmay : M-length array
			standard deviation of y
		p0 : N-length sequence
			initial guess for parameters
		print_info : bool, optional
			If True, print information about the fit
		absolute_sigma : bool, optional
			If False, compute asymptotic errors, else standard errors for parameters

		Returns
		-------
		par : N-length array
			optimal values for parameters
		cov : (N,N)-shaped array
			covariance matrix of par

		Notes
		-----
		This is a wrapper of scipy.odr
	"""
	f_wrap = lambda params, x: f(x, *params)
	model = odr.Model(f_wrap)
	data = odr.RealData(x, y, sx=sigmax, sy=sigmay)
	ODR = odr.ODR(data, model, beta0=p0)
	output = ODR.run()
	par = output.beta
	cov = output.cov_beta
	if print_info:
		output.pprint()
	if (not absolute_sigma) and len(y) > len(p0):
		s_sq = sum(((np.asarray(y) - f(x, *par)) / (np.asarray(sigmay))) ** 2) / (len(y) - len(p0))
		cov *= s_sq
	return par, cov

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

def _fit_linear_yerr(x, y, sigmay):
	dy2 = sigmay ** 2
	sx2 = (x ** 2 / dy2).sum()
	sxy = (x * y / dy2).sum()
	a = sxy / sx2
	b = 0
	vaa = 1 / sx2
	vbb = 0
	vab = 0
	return np.array([a, b]), np.array([[vaa, vab], [vab, vbb]])

def _fit_linear_unif_err(x, y):
	sx2 = (x ** 2).sum()
	sxy = (x * y).sum()
	a = sxy / sx2
	b = 0
	vaa = 1 / sx2
	vbb = 0
	vab = 0
	return np.array([a, b]), np.array([[vaa, vab], [vab, vbb]])

def fit_linear(x, y, dx=None, dy=None, offset=True, absolute_sigma=True, conv_diff=0.001, max_cycles=5, print_info=False):
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
		fun_fit = _fit_affine_yerr
		fun_fit_dynone = _fit_affine_unif_err
	else:
		fun_fit = _fit_linear_yerr
		fun_fit_dynone = _fit_linear_unif_err
	if not (dy is None):
		dy = np.asarray(dy)
		par, cov = fun_fit(x, y, dy)
		if not absolute_sigma:
			chisq_rid = (((y - par[0]*x - par[1]) / dy)**2).sum() / (len(x) - 2)
			cov *= chisq_rid
	else:
		par, cov = fun_fit_dynone(x, y)
		chisq_rid = ((y - par[0]*x - par[1])**2).sum() / (len(x) - 2)
		cov *= chisq_rid
		dy = 0
	if dx is None:
		return par, cov
	dx = np.asarray(dx)
	cycles = 1
	while True:
		if cycles >= max_cycles:
			raise RuntimeError("Maximum number of fit cycles %d reached" % max_cycles)
		dyeff = np.sqrt(dy**2 + (par[0] * dx)**2)
		npar, cov = fun_fit(x, y, dyeff)
		error = abs(npar - par)
		par = npar
		if not absolute_sigma:
			chisq_rid = (((y - par[0]*x - par[1]) / dyeff)**2).sum() / (len(x) - 2)
			cov *= chisq_rid
		cycles += 1
		if all(error <= np.sqrt(np.diag(cov)) * conv_diff):
			break
	if print_info:
		print(fit_linear, ": cycles: %d" % (cycles))
	return par, cov

def fit_affine_noerr(x, y):
	"""
		fit y = a * x + b

		Parameters
		----------
		x : M-length array
			independent data
		y : M-length array
			dependent data

		Returns
		-------
		a : float
			optimal value for a
		b : float
			optimal value for b
	"""
	x = np.asarray(x)
	y = np.asarray(y)
	sy = math.fsum(y)
	sx2 = math.fsum(x ** 2)
	sx = math.fsum(x)
	sxy = math.fsum(x * y)
	denom = len(x) * sx2 - sx ** 2
	a = (len(x) * sxy  - sx * sy) / denom
	b = (sy * sx2 - sx * sxy) / denom
	return np.array([a, b])

def fit_affine_xerr(x, y, sigmax):
	"""
	fit y = m * x + q

	Parameters
	----------
	x : M-length array
		independent data
	y : M-length array
		dependent data
	sigmax : M-length array
		standard deviation of x

	Returns
	-------
	par:
		estimates (m, q)
	cov:
		covariance matrix m,q

	Notes
	-----
	Implementation: consider the inverse relation:
		x = 1/m * y - q/m
	find 1/m and -q/m using fit_linear then compute m, q and their variances
	with first-order error propagation.
	"""
	x = np.asarray(x)
	y = np.asarray(y)
	sigmax = np.asarray(sigmax)
	par, cov = _fit_affine_yerr(y, x, sigmax)
	m, q = par
	dmm, dmq, _, dqq = cov.flat
	a = 1 / m
	b = -q / m
	daa = a**2 * (dmm/m**2)
	dbb = b**2 * (dqq/q**2 + dmm/m**2 + 2*dmq/(-q*m))
	dab = dmm*(-1/m**2)*(q/m**2) + dmq*(-1/m**2 * -1/m)
	return np.array([a, b]), np.array([[daa, dab], [dab, dbb]])

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
			relres=[50] * 7,
			valg=[1] * 7
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
			scales=[10*5e-10] + [ (10*d*10**s) for s in range(-9, 2) for d in [1, 2.5, 5] ],
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
	sqerr : bool
		If True, sum errors squaring.

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

	errsum = (lambda x, y: math.sqrt(x**2 + y**2)) if sqerr else (lambda x, y: x + y)

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
			r = 10e+6
		elif unit == 'ampere' or unit == 'ampere_ac':
			r = 0.2 / s
	elif typ == 'analog':
		e = x * errsum(0.5 / info['relres'][idx], info['valg'][idx] / 100.0 * s)
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
	determines the fullscale used to measure x with a multimeter,
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

# *********************** FORMATTING *************************

d = lambda x, n: int(("%.*e" % (n - 1, abs(x)))[0])
ap = lambda x, n: float("%.*e" % (n - 1, x))
nd = lambda x: math.floor(math.log10(abs(x))) + 1
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
	dn = int(nd(x) - nd(e)) if x != 0 else -n
	nx = n + dn
	if nx > 0:
		ex = nd(x) - 1
		if nx > ex and abs(ex) <= minexp:
			xd = nx - ex - 1
			ex = 0
		else:
			xd = nx - 1
		sx = "%.*f" % (xd, x / 10**ex)
		se = "%.*f" % (xd, e / 10**ex)
	else:
		le = nd(e)
		ex = le - n
		sx = '0'
		se = "%#.*g" % (n, e)
	# RETURN
	if errsep:
		return sx, se, ex
	return sx + '(' + ("%#.*g" % (n, e * 10 ** (n - nd(e))))[:n] + ')', '', ex

def util_format(x, e, pm=None, percent=False, comexp=True):
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
	util_format(nan, nan) = 'nan +- nan'

	See also
	--------
	xe, xep
	"""
	x = float(x)
	e = abs(float(e))
	if not math.isfinite(x) or not math.isfinite(e) or e == 0:
		return "%.3g %s %.3g" % (x, '+-', e)
	sx, se, ex = _format_epositive(x, e, not (pm is None))
	es = "e%+d" % ex if ex != 0 else ''
	if pm is None:
		s = sx + es
	elif comexp and es != '':
		s = '(' + sx + ' ' + pm + ' ' + se + ')' + es
	else:
		s = sx + es + ' ' + pm + ' ' + se + es
	if (not percent) or sx == '0':
		return s
	pe = e / x * 100.0
	return s + " (%.*g %%)" % (2 if pe < 100.0 else 3, pe)

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

_eta_start = 0

def etastart():
	"""
	Call at the startpoint of something you want to compute the eta (estimated
	time of arrival) of.

	Returns
	-------
	An object containing the starting time, to be given as argument to etastr().

	Example
	-------
	>>> eta = etastart()
	>>> for i in range(N):
	>>>     print('elapsed time: %s, remaining time: %s' % etastr(eta, i / N))
	>>>     # do something

	See also
	--------
	etastr
	"""
	return time.time()

def etastr(eta, progress):
	"""
	Compute the eta given a startpoint returned from etastart() and the progress.

	Parameters
	----------
	eta :
		object returned by etastart()
	progress : number in [0,1]
		the progress on a time-linear scale where 0 means still nothing done and
		1 means finished.

	Returns
	-------
	timestr : string
		elapsed time
	etastr : string
		estimated time remaining

	Example
	-------
	>>> eta = etastart()
	>>> for i in range(N):
	>>>     print('elapsed time: %s, remaining time: %s' % etastr(eta, i / N))
	>>>     # do something

	See also
	--------
	etastart
	"""
	interval = time.time() - eta
	if 0 < progress <= 1:
		etastr = util_timestr((1 - progress) * interval / progress)
	elif progress == 0:
		etastr = "--:--:--"
	else:
		raise RuntimeError("progress %.2f out of bounds [0,1]" % progress)
	timestr = util_timestr(interval)
	return timestr, etastr

# this function taken from stackoverflow and modified
# http://stackoverflow.com/questions/17973278/python-decimal-engineering-notation-for-mili-10e-3-and-micro-10e-6
def num2si(x, format='%.16g', si=True, space=' '):
	"""
	Returns x formatted in a simplified engineering format -
	using an exponent that is a multiple of 3.

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
	util_format, util_format_comp, xe, xep
	"""
	x = float(x)
	if x == 0:
		return format % x
	exp = int(math.floor(math.log10(abs(x))))
	exp3 = exp - (exp % 3)
	x3 = x / (10 ** exp3)

	if si and exp3 >= -24 and exp3 <= 24 and exp3 != 0:
		exp3_text = 'yzafpnum kMGTPEZY'[(exp3 - (-24)) // 3]
	elif exp3 == 0:
		exp3_text = ''
		space = ''
	else:
		exp3_text = 'e%s' % exp3

	return (format + '%s%s') % (x3, space, exp3_text)

# ************************ SHORTCUTS ******************************

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

_util_format_vect = np.vectorize(util_format, otypes=[str])

unicode_pm = u'±'

def xe(x, e, pm=None, comexp=True):
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
	return _util_format_vect(x, e, pm, False, comexp)

def xep(x, e, pm=None, comexp=True):
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
	return _util_format_vect(x, e, pm, True, comexp)
