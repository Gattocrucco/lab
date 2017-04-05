from lab import *
from lab import _Nonedict
from scipy import stats, linalg
import time
import sympy
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

#### PARAMETERS ####

showplot = True # show plot after monte carlo run with fixed parameters
showpsplot = False # show parameter biases with fixed data
showpsdtplot = False # show parameter vs. errors

p0s = [ # true parameters, axis 0 = parameter, axis 1 = values
	# linspace(-1,1,10),
	# logspace(0,1,10),
	[1],
	[1],
#	[1],
]
fs = [ # sympy functions
	lambda x, a, b: a * sympy.exp(x / b),
	lambda x, m, q: m * x + q,
	lambda x, m: m * x,
	lambda t, A, w, phi: A * sympy.sin(w * t + phi)
]
f = fs[1] # function to fit

mcn = 1000 # number of repetitions (monte carlo)
method = 'ml' # ev, linodr, odrpack, ml, wleastsq, leastsq
xmean = np.linspace(0, 10, 100) # true x
n = len(xmean) # number of points
dys = np.outer([1], np.ones(n)*.1) # errors, axis 0 = dataset, axis 1 = point
dxs = np.outer([1], np.ones(n)*.1)
####################

# initialize symbols
psym = [sympy.Symbol('p_%d' % i, real=True) for i in range(len(p0s))]
xsym = sympy.Symbol('x', real=True)
syms = [xsym] + psym

# format function in LaTeX and 1D text
flatex = sympy.latex(f(*syms))
psubsym = [sympy.Symbol('p%s' % num2sub(i), real=True) for i in range(len(p0s))]
fstr = str(f(xsym, *psubsym)).replace('**', '^').replace('*', '·')

model = FitModel(f, symb=True)

fsym = f
f = sympy.lambdify(syms, f(*syms), "numpy")

# initialize output arrays
p0shape = [len(p0) for p0 in p0s]
fp = np.empty([len(dxs), len(dys)] + p0shape + [len(p0s)]) # fitted parameters (mean over MC)
cp = np.empty([len(dxs), len(dys)] + p0shape + 2 * [len(p0s)]) # fitted parameters mean covariance matrices
chisq = np.empty(mcn) # chisquares from 1 MC run
pars = np.empty((mcn, len(p0s))) # parameters from 1 MC run
covs = np.empty((mcn, len(p0s), len(p0s))) # covariance matrices from 1 MC run
times = np.empty(mcn) # execution times from 1 MC run

def plot_text(string, loc=2, ax=plt.gca(), **kw):
	locs = [
		[],
		[.95, .95, 'right', 'top'],
		[.05, .95, 'left', 'top']
	]
	loc = locs[loc]
	ax.text(loc[0], loc[1], string, horizontalalignment=loc[2], verticalalignment=loc[3], transform=ax.transAxes, **kw)
			
eta = Eta()
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
				
				# compute progress
				progress = l + len(dxs) * ll
				for j in range(len(K)):
					progress *= p0shape[j]
					progress += K[j]
				progress *= mcn
				progress += i
				progress /= len(dxs) * len(dys) * np.prod(p0shape) * mcn
				eta.etaprint(progress)
				
				# generate data
				deltax = stats.norm.rvs(size=n)
				x = xmean + dx * deltax
				deltay = stats.norm.rvs(size=n)
				y = ymean + dy * deltay
				
				# fit
				start = time.time()
				out = fit_generic(model, x, y, dx, dy, p0=p0, method=method, **_Nonedict(max_cycles=10 if method == 'ev' else None))
				end = time.time()
				
				# save results
				if showplot:
					times[i] = end - start
					chisq[i] = out.chisq
				pars[i] = out.par
				covs[i] = out.cov
			
			# save results
			icovs = np.empty(covs.shape)
			for i in range(len(icovs)):
				icovs[i] = linalg.inv(covs[i])
			pc = linalg.inv(icovs.sum(axis=0))
			wpar = np.empty(pars.shape)
			for i in range(len(wpar)):
				wpar[i] = icovs[i].dot(pars[i])
			pm = pc.dot(wpar.sum(axis=0))
			ps = np.sqrt(np.diag(pc))

			fp[(l, ll) + K] = pm
			cp[(l, ll) + K] = pc
			
			if showplot:
				
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
				
				fig = plt.figure('Function %s, fit with method “%s”' % (fstr, method), figsize=(4*cols, 2.3*rows))
				fig.clf()
				fig.set_tight_layout(True)
				
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
				ax.plot(fx, f(fx, *pars[-1]), 'r-', linewidth=1, label='Fit', zorder=3, alpha=1)
				# plot_text('$y=%s$\n$y=%s$' % (flatex, sympy.latex(fsym(xsym, *p0))), fontsize=20, ax=ax)
				ax.ticklabel_format(style='sci', axis='both', scilimits=(-3,3))
				ax.legend(loc=0, fontsize='small')
				
				# save figure and show
				fig.savefig(nextfilename(fig.canvas.get_window_title(), '.pdf', prepath='Figures'))
				fig.show()

if showpsplot:
	fig = plt.figure('Function %s, fit with method “%s”, parameter biases' % (fstr, method), figsize=(10,7))
	fig.clf()
	fig.set_tight_layout(True)
	for i in range(len(p0s)): # p_i = fitted
		for j in range(len(p0s)): # p_j = true
			ax = fig.add_subplot(len(p0s), len(p0s), 1 + i * len(p0s) + j)
			K = [0] * len(p0s)
			K[j] = Ellipsis
			K = tuple(K)
			ax.errorbar(p0s[j], fp[(0, 0) + K + (i,)] - (np.asarray(p0s[i]) if i == j else p0s[i][0]), sqrt(cp[(0, 0) + K + (i, i)]), fmt=',')
			ax.set_xlabel('True $p_{%d}$' % j)
			ax.set_ylabel('$p_{%d}\'-p_{%d}$' % (i, i))
			pstr = ''
			for k in range(len(p0s)):
				if k != j:
					pstr += '$p_{%d}$ = %.2g\n' % (k, p0s[k][0])
			plot_text(pstr, ax=ax)
			ax.ticklabel_format(style='sci', axis='both', scilimits=(-3,3))
			ax.grid()
				
	fig.savefig(nextfilename(fig.canvas.get_window_title(), '.pdf', prepath='Figures'))
	fig.show()
	# figure('Slope 3d, fit with method “%s”' % method).set_tight_layout(True)
	# clf()
	# subplot(111, projection='3d')
	# X, Y = meshgrid(ms, qs)
	# gca().plot_surface(X, Y, (fp[:,:,0,0,0]-ms[:,newaxis]).T, rstride=1, cstride=1)
	# xlabel('True m')
	# ylabel('True q')
	# gca().set_zlabel('$m\'-m$')

if showpsdtplot:
	
	fig = plt.figure('Function %s, fit with method “%s”, parameters vs. errors' % (fstr, method), figsize=(10,7))
	fig.clf()
	fig.set_tight_layout(True)
	
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
		
	fig.savefig(nextfilename(fig.canvas.get_window_title(), '.pdf', prepath='Figures'))
	fig.show()
