from lab import *
import sympy
import numpy as np

#### PARAMETERS ####
showplot = True # show plot after monte carlo run with fixed parameters
showpsplot = False # show parameter biases with fixed data
showpsdtplot = False # show parameter vs. errors

p0s = [ # true parameters, axis 0 = parameter, axis 1 = values
	# linspace(-1,1,10),
	# logspace(0,1,10),
	[1],
	[2],
    # [3],
]
fs = [ # sympy functions
	lambda x, a, b: a * sympy.exp(x / b),
	lambda x, m, q: m * x + q,
	lambda y, m, q: (y - q) / m,
	lambda x, theta, b: (x * sympy.sin(theta) + b) / sympy.cos(theta),
	lambda x, m: m * x,
	lambda t, A, w, phi: A * sympy.sin(w * t + phi)
]
f = fs[1] # function to fit
bounds = [
    [0, 0],
    [3, 3]
]
# bounds=None

mcn = 2 # number of repetitions (monte carlo)
methods = ['pymc3'] # ev, linodr, odrpack, ml, wleastsq, leastsq, pymc3
xmean = np.linspace(0, 10, 20) # true x
n = len(xmean) # number of points
dys = np.outer([1], np.ones(n)*.1) # errors, axis 0 = dataset, axis 1 = point
dxs = np.outer([1], np.ones(n)*.1)
####################

method_kw = []
for m in methods:
	if m == 'ev':
		d = dict(max_cycles=50)
	elif m == 'pymc3':
		d = dict(nsamples=1000, print_info=1, init=True, njobs=2)
	else:
		d = dict()
	method_kw.append(d)

model = CurveModel(f, symb=True)
plot = dict(single=showplot, vsp0=showpsplot, vsds=showpsdtplot)
out = fit_curve_bootstrap(model, xmean, dxs=dxs, dys=dys, p0s=p0s, mcn=mcn, method=methods, plot=plot, eta=True, wavg=False, sdmean=False, method_kw=method_kw, full_output=True, bounds=bounds)

figs = []
if showplot:
	figs.append(out.plot['single'][-1])
if showpsplot:
	figs.append(out.plot['vsp0'])
if showpsdtplot:
	figs.append(out.plot['vsds'])
for fig in figs:
	fig.savefig(nextfilename(fig.canvas.get_window_title(), '.pdf', prepath='Figures'))
	fig.show()

data_file = nextfilename(str(model) + ' method(s) ' + ', '.join(methods), '.npz', prepath='Data')
np.savez_compressed(data_file, fp_mc=out.fp_mc, cp_mc=out.cp_mc)
