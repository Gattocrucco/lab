
from itertools import count as itercount
import numpy as np
import scipy.stats.distributions as dists
import matplotlib.pyplot as plt
import matplotlib as mpl
import inspect
from lab import *

# *********************** UTILITIES *************************


def parallel(*Zn):
	"""Calculate impedance parallel."""
	invZ = 0
	for Zi in Zn:
		if np.abs(Zi) < np.inf:
			invZ += 1/Zi
	return 1/invZ


def lineheight(linepars, linecov, height, hvar=0):
	"""Find intersection (with variance) of straight line and constant height."""
	m, q = linepars
	h = height
	x = (h - q) / m
	derivs = [(q - h)/m**2, -1/m, 1/m]
	covmat = np.zeros((3, 3))
	covmat[:2, :2] = linecov
	covmat[2, 2] = hvar
	varx = derivs @ covmat @ derivs
	return x, varx


def tell_chi2(resd, dofs, style='normal'):
	"""
	Chi^2 prettyprinter.

	Calculate chi squared from normalized redisuals and return
	a string telling it along with p-value and degrees of freedom.

	Parameters
	----------
	resd: (iterable of) number
		the normalised residuals of a fitted dataset
	dofs: unsigned integer
		the number of degrees of freedom of the fit
	style: stirng, optional (default 'normal')
		if 'latex', the string will be formatted with LaTeX-like symbols macro

	Returns
	-------
	chi2msg: string
		the message printed
	"""
	chi2 = np.sum(resd**2)
	pval = dists.chi2.sf(chi2, dofs)
	if style == 'normal':
		chiname = R"ChiSquare"
		dofname = R"DoFs,"
	elif style == 'latex':
		chiname = R"\chi^2"
		dofname = R"\dof , \ "
	chi2msg = "{0} = {1:.2f} ({2} {3} p = {4:.4f})"
	return chi2msg.format(chiname, chi2, dofs, dofname, pval)


def maketab(*columns, errors='all', precision=3):
	"""
	Print data in tabular form.

	Creates a LaTeX (table environment with S columns from siunitx) formatted
	table using inputs as columns; the input is assumed to be numerical.
	Errors can be given as columns following the relevant numbers column
	and their positions specified in a tuple given as the kwarg 'errors';
	they will be nicely formatted for siunitx use.
	The precision (number of signifitant digits) of the numbers representation
	will be inferred from errors, or when absent from the 'precision' kwarg.

	Parameters
	----------
	*columns: N iterables of lenght L with numerical items
		the lists of numbers that will make up the columns of the table,
		must be of uniform lenght.
	errors: tuple(-like) of ints, 'all' or 'none', optional (default 'all')
		the columns with positions corresponging to errors items will
		be considered errors corresponding to the previous column and formatted
		accordingly; if 'all' every other column will be considered an error,
		if 'none' no column will be considered an error.
	precision: (lenght-L tuple of) int, optional (default 3)
		number of significant digits to be used when error is not given;
		if a tuple(-like) is given, each column will use the correspondingly
		indexed precision.

	Returns
	-------
	tab: string
		the formatted text constituting the LaTeX table.
	"""
	vals = np.asarray(columns).T
	cols = np.alen(vals.T)
	precision = np.array(precision) * np.ones(cols)
	if errors == 'all':
		errors = range(1, cols, 2)
	if errors == 'none':
		errors = []
	beginning = (
		R"\begin{table}" "\n\t"
		R"\begin{tabular}{*{"
		+ str(cols - len(errors)) +
		R"}{S}}"
		"\n\t\t"
		R"\midrule" "\n"
	)
	inner = ""
	for i, row in enumerate(vals):
		rows = enumerate(row, start=1)
		inner += "\t"
		for pos, v in rows:
			num = v if np.isfinite(v) else "{-}"
			space = "&" if pos < cols else "\\\\ \n"
			err = ""
			prec = -precision[pos-1]
			if pos in errors:
				prec = np.floor(np.log10(row[pos]))
				if row[pos]/10**prec < 2.5:
					prec -= 1
				err = "({0:.0f})".format(round(row[pos]/10**prec))
				if next(rows, (-1, None))[0] >= cols:
					space = "\\\\ \n"
				num = round(num, int(-prec))
			inner += "\t{0:.{digits:.0f}f} {1}\t{2}".format(num, err, space, digits=max(0, -prec))
	ending = (
		"\t" R"\end{tabular}" "\n"
		"\t" R"\caption{some caption}" "\n"
		R"\label{t:somelabel}" "\n"
		R"\end{table}"
	)
	return beginning + inner + ending

# *********************** FACTORIES *************************


_dline = np.vectorize(lambda x, m, q: m)
_dlogline = np.vectorize(lambda x, m, q: m*np.log10(np.e)/x)
_nullfunc = np.vectorize(lambda *args: 0)
_const = np.vectorize(lambda x, q: q)


def createline(type='linear', name=""):
	"""
	Factory of linear(-like) functions.

	Parameters
	----------
	type: str, optional (default 'linear')
		if 'log', returns a function linear in log10(x).
	name: str, optional
		if provided, the function returned will have this name,
		otherwise it will be named "line" (or "logline" if type is 'log').

	Returns
	-------
	func: callable
		func(x, m, q) = mx + q (or m*log10(x) + q if type is 'log').
	"""
	if type == 'log':
		def logline(x, m, q):
			"""f: (x, m, q) --> m*log10(x) + q ."""
			return m*np.log10(x) + q
		logline.deriv = _dlogline
		func = logline
	elif type == 'linear':
		def line(x, m, q):
			"""f: (x, m, q) --> m*x + q ."""
			return m*x + q
		line.deriv = _dline
		func = line
	elif type == 'const':
		_h = np.vectorize(lambda x, q: q)

		def flatline(x, q):
			"""f: (x, q) --> q ."""
			return _h(x, q)
		flatline.deriv = _nullfunc
		func = flatline
	if name:
		func.__name__ = name
	return func

# *********************** OBJECTS *************************


class MeasureObj(object):

	def __init__(self, val, err=0):
		self.val = np.array(val)
		self.err = np.array(err) * np.ones(len(self.val))


class DataHolder(object):

	_holdercount = itercount(1)

	def __init__(self, x, y, dx=0, dy=0, name=None):
		self.num = next(self._holdercount)
		self.fig = plt.figure(self.num)
		self.datakw = dict(fmt='none', ecolor='black', label='data')
		self.pts = None
		self.title = ""
		if name:
			self.name = name
		else:
			self.name = "dataset #{}".format(self.num)

		self.x = MeasureObj(x, dx)
		self.x.edge_padding = 1/20
		self.y = MeasureObj(y, dy)
		self.y.edge_padding = 3/20

		for z in (self.x, self.y):
			z.lims = None
			z.type = 'linear'
			z.re = 1
			z.label = ""

	def fit_generic(self, *funcs, verbose=True, **kwargs):

		if verbose:
			print("Lavoro su {}\n".format(self.name))
			fitmsg = "Il fit di {funname} su {dataname} ha dato i seguenti parametri:"
		for f in funcs:
			mask = getattr(f, 'mask', np.ones(len(self.x.val), dtype=bool))
			x = self.x.val[mask]
			y = self.y.val[mask]
			dx = self.x.err[mask]
			dy = self.y.err[mask]
			p0 = getattr(f, 'pars', None)
			df = getattr(f, 'deriv', _nullfunc)
			pars, pcov = fit_generic(f, x, y, dx, dy, p0=p0, **kwargs)
			f.pars = pars
			f.cov = pcov
			f.sigmas = np.sqrt(np.diag(pcov))
			f.resd = (y - f(x, *pars)) / np.sqrt(dy**2 + dx**2 * df(x, *pars)**2)
			if verbose:
				print(fitmsg.format(dataname=self.name, funname=f.__name__))
				argnames = inspect.getargspec(f).args[1:]
				for name, par, err in zip(argnames, pars, f.sigmas):
					print("{0} = {1:.4f} \pm {2:.4f}".format(name, par, err))
				tell_chi2(f.resd, np.alen(x) - len(pars), style='latex')
				print("")
			if verbose:
				print("{} completo\n\n".format(self.name))

	def _set_edges(self, var, type=None):
		if var == 'x':
			z = self.x
		elif var == 'y':
			z = self.y
		if not type:
			type = z.type
		top = np.amax(z.val)
		bot = np.amin(z.val)
		if type == 'log':
			top = np.log10(top)
			bot = np.log10(bot)
		width = top - bot
		high = top + width * z.edge_padding
		low = bot - width * z.edge_padding
		if type == 'log':
			high = 10**high
			low = 10**low
		z.lims = np.array([low, high])

	def _getpts(self, type=None):
		if not type:
			type = self.x.type
		low = self.x.lims[0]
		high = self.x.lims[1]
		if type == 'log':
			self.pts = np.logspace(np.log10(low), np.log10(high), num=max(len(self.x.val)*10, 200))
		elif type == 'linear':
			self.pts = np.linspace(low, high, num=max(len(self.x.val)*10, 200))

	def _graph_setup(self, resid=False):
		if not self.x.lims:
			self._set_edges('x')
		if not self.y.lims:
			self._set_edges('y')
		if not self.pts:
			self._getpts()
		if not resid:
			main_ax = self.fig.add_subplot(1, 1, 1)
			main_ax.set_xlabel(self.x.label)
			resid = ()
		else:
			sub_gs = mpl.gridspec.GridSpec(5, 1)		# TODO: make better
			main_ax = self.fig.add_subplot(sub_gs[:4])
			resd_ax = self.fig.add_subplot(sub_gs[4:])
			resd_ax.axhline(y=0, color='black')
			resd_ax.set_xlabel(self.x.label)
			resd_ax.set_xscale(self.x.type)
			resd_ax.set_xlim(*(self.x.lims * self.x.re))
			self.resd_ax = resd_ax

		main_ax.set_xlim(*(self.x.lims * self.x.re))
		main_ax.set_ylim(*(self.y.lims * self.y.re))
		main_ax.set_ylabel(self.y.label)
		main_ax.set_title(self.title)
		main_ax.set_xscale(self.x.type)
		main_ax.set_yscale(self.y.type)
		self.main_ax = main_ax

	def draw(self, *funcs, resid=False, data=True, legend=True):
		self._graph_setup(resid)
		main_ax = self.main_ax
		if not resid:
			resid = ()
		else:
			resd_ax = self.resd_ax
			if resid is True:
				resid = funcs

		x = self.x.val * self.x.re
		y = self.y.val * self.y.re
		dx = self.x.err * self.x.re
		dy = self.y.err * self.y.re
		if data:
			main_ax.errorbar(x, y, dy, dx, **self.datakw)

		for fun in funcs:
			if callable(fun):
				mask = np.zeros(len(self.pts), dtype=bool)
				for lowest, highest in getattr(fun, 'bounds', [(-np.inf, np.inf)]):
					mask |= (self.pts > lowest) & (self.pts < highest)
				points = self.pts[mask]
				try:
					linekw = fun.linekw
				except AttributeError:
					linekw = fun.linekw = {}
				g, = main_ax.plot(points * self.x.re, fun(points, *fun.pars)*self.y.re, **linekw)
				if 'color' not in linekw:
					linekw['color'] = g.get_color()
			else:
				pass

		if legend:
			main_ax.legend(loc='best')

		for fun in resid:
			mask = getattr(fun, 'mask', np.ones(len(x), dtype=bool))
			resdkw = dict(marker='o')
			if hasattr(fun, 'linekw'):
				resdkw.update(fun.linekw)
			resdkw.update(ls='none')
			if hasattr(fun, 'resd'):
				res = fun.resd
			else:
				df = getattr(fun, 'deriv', _nullfunc)
				delta = self.y.val - fun(self.x.val, *fun.pars)
				variance = self.y.err**2 + self.x.err**2 * df(self.x.val, *fun.pars)**2
				fun.resd = res = delta / np.sqrt(variance)
			resd_ax.plot(x[mask], res, **resdkw)
