import lab
import scipy.stats as st
from pylab import *
from scipy.optimize import curve_fit
import time
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D
from strangefit import *
from scipy.optimize import curve_fit
import os

#### PARAMETERS ####

showplot = True # show plot after monte carlo run with fixed parameters
showpsplot = False # show parameter biases with fixed data
showpsdtplot = False # show parameter vs. errors
stattest = False # perform statistical test after monte carlo

p0s = [ # true parameters, axis 0 = parameter, axis 1 = values
	# linspace(-1,1,10),
	# logspace(0,1,10),
	[1],
	[1],
	[1],
]
fs = [ # sympy functions
	lambda x, a, b: a * sp.exp(x / b),
	lambda x, m, q: m * x + q,
	lambda x, m: m * x,
	lambda t, A, w, phi: A * sp.sin(w * t + phi)
]
f = fs[3] # function to fit

mcn = 1000 # number of repetitions (monte carlo)
fitfun = 'odrpack' # ev, odr, odrpack, curve_fit, hoch
xmean = linspace(0, 10, 100) # true x
n = len(xmean) # number of points
dys = outer([1], ones(n)*.1) # errors, axis 0 = dataset, axis 1 = point
dxs = outer([1], ones(n)*.1)
####################

# initialize symbols
psym = [sp.Symbol('p_%d' % i, real=True) for i in range(len(p0s))]
xsym = sp.Symbol('x', real=True)
syms = [xsym] + psym

# format function in LaTeX and 1D text
flatex = sp.latex(f(*syms))
psubsym = [sp.Symbol('p%s' % lab.num2sub(i), real=True) for i in range(len(p0s))]
fstr = str(f(xsym, *psubsym)).replace('**', '^').replace('*', '·')

model = lab.FitModel(f)

fsym = f
f = sp.lambdify(syms, f(*syms), "numpy")

# initialize output arrays
p0shape = [len(p0) for p0 in p0s]
fp = empty([len(dxs), len(dys)] + p0shape + [len(p0s)]) # fitted parameters (mean over MC)
cp = empty([len(dxs), len(dys)] + p0shape + 2 * [len(p0s)]) # fitted parameters mean covariance matrices
chisq = empty(mcn) # chisquares
pars = empty((mcn, len(p0s))) # parameters from 1 MC run
covs = empty((mcn, len(p0s), len(p0s))) # covariance matrices from 1 MC run
times = empty(mcn) # execution times from 1 MC run

def plot_text(string, loc=2, **kw):
	locs = [
		[],
		[.95, .95, 'right', 'top'],
		[.05, .95, 'left', 'top']
	]
	loc = locs[loc]
	text(loc[0], loc[1], string, horizontalalignment=loc[2], verticalalignment=loc[3], transform=gca().transAxes, **kw)
			
eta = lab.etastart()
for ll in range(len(dys)):
	dy = dys[ll]
	for l in range(len(dxs)):
		dx = dxs[l]
		for K in ndindex(*p0shape):
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
				progress /= len(dxs) * len(dys) * prod(p0shape) * mcn
				lab.etastr(eta, progress, mininterval=5)
				
				# generate data
				deltax = st.norm.rvs(size=n)
				x = xmean + dx * deltax
				deltay = st.norm.rvs(size=n)
				y = ymean + dy * deltay
				
				# fit
				start = time.time()
				if fitfun == 'ev':
					par, cov = lab.fit_generic(model, x, y, dx, dy, p0=p0, method='ev', max_cycles=10)
				elif fitfun == 'odr':
					par, cov = lab.fit_generic(model, x, y, dx, dy, p0=p0, method='linodr')
				elif fitfun == 'hoch':
					par, cov = fit_generic_xyerr4(model.f(), model.fi(), model.dfdp_odrpack(), model.dfidp(), x, y, dx, dy, p0)
				elif fitfun == 'odrpack':
					par, cov = lab.fit_generic(model, x, y, dx, dy, p0=p0, method='odrpack')
				elif fitfun == 'curve_fit':
					par, cov = curve_fit(f, x, y, sigma=dy, p0=p0, absolute_sigma=True, jac=model.dfdp_curve_fit(len(x)))
				end = time.time()
				
				# save results
				if showplot or stattest:
					times[i] = end - start
					chisq[i] = ((y - model.f()(x, *par))**2 / (dy**2 + (model.dfdx()(x, *par)*dx)**2)).sum()
				pars[i] = par
				covs[i] = cov
			
			# save results
			icovs = empty(covs.shape)
			for i in range(len(icovs)):
				icovs[i] = np.linalg.inv(covs[i])
			pc = np.linalg.inv(icovs.sum(axis=0))
			wpar = empty(pars.shape)
			for i in range(len(wpar)):
				wpar[i] = icovs[i].dot(pars[i])
			pm = pc.dot(wpar.sum(axis=0))
			ps = sqrt(diag(pc))

			fp[(l, ll) + K] = pm
			cp[(l, ll) + K] = pc
			
			if stattest or showplot:
				
				prho = pc / outer(ps, ps)

				pdist = (pm - array(p0)) / ps
				pdistc = (outer(pm - array(p0), pm - array(p0)) - pc) / sqrt(pc**2 + outer(ps, ps)**2)
				chidist = (chisq.mean() - (n-len(p0))) / chisq.std(ddof=1) * sqrt(len(chisq))
				
				pvalue = st.kstest(chisq, 'chi2', (n-len(p0),))[1]
				
				print(pdist, chidist, pvalue)
			
			if showplot:
				
				maxscatter = 1000
				histkw = dict(
					bins=int(sqrt(min(mcn, 1000))),
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
				
				figure('Function %s, fit with method “%s”' % (fstr, fitfun), figsize=(5*cols,3*rows)).set_tight_layout(True)
				clf()
				
				# histogram of parameter; diagonal
				for i in range(len(p0)):					
					subplot(rows, cols, 1 + i * (1 + cols))
					title("$(p_{%d}'-{p}_{%d})/\sigma_{%d}$" % (i, i, i))
					S = (pars[:,i] - p0[i]) / sqrt(covs[:,i,i])
					hist(S, **histkw)
					plot_text("$p_%d = $%g\n$\\bar{p}_{%d}-p_{%d} = $%.2g $\\bar{\sigma}_{%d}$" % (i, p0[i], i, i, pdist[i], i))
					ticklabel_format(style='sci', axis='x', scilimits=(-3,3))
					
				# histogram of covariance; lower triangle
				for i in range(len(p0)):
					for j in range(i):
						subplot(rows, cols, 1 + i * cols + j)
						title("$((p_{%d}'-p_{%d})\cdot(p_{%d}'-p_{%d})-\sigma_{%d%d})/\sqrt{\sigma_{%d%d}^2+\sigma_{%d}^2\sigma_{%d}^2}$" % (i, i, j, j, i, j, i, j, i, j))
						C = ((pars[:,i] - p0[i]) * (pars[:,j] - p0[j]) - covs[:,i,j]) / sqrt(covs[:,i,j]**2 + covs[:,i,i]*covs[:,j,j])
						hist(C, **histkw)
						plot_text("$\\bar{\\rho}_{%d%d} = $%.2g\n$(\\bar{p}_{%d}-p_{%d})\cdot(\\bar{p}_{%d}-p_{%d})-\\bar{\sigma}_{%d%d} = $%.2g $\sqrt{\\bar{\sigma}_{%d%d}^2+\\bar{\sigma}_{%d}^2\\bar{\sigma}_{%d}^2}$" % (i, j, prho[i,j], i, i, j, j, i, j, pdistc[i,j], i, j, i, j))
						ticklabel_format(style='sci', axis='x', scilimits=(-3,3))
				
				# scatter plot of pairs of parameters; upper triangle
				for i in range(len(p0)):
					for j in range(i + 1, len(p0)):
						subplot(rows, cols, 1 + i * cols + j)
						title("$(p_{%d}'-p_{%d})/\sigma_{%d}$, $(p_{%d}'-p_{%d})/\sigma_{%d}$" % (i, i, i, j, j, j))
						X = (pars[:, i] - p0[i]) / sqrt(covs[:,i,i])
						Y = (pars[:, j] - p0[j]) / sqrt(covs[:,j,j])
						if len(X) > maxscatter:
							X = X[::int(ceil(len(X) / maxscatter))]
							Y = Y[::int(ceil(len(Y) / maxscatter))]
						plot(X, Y, '.k', markersize=2)
						grid()
						ticklabel_format(style='sci', axis='both', scilimits=(-3,3))
				
				# histogram of chisquare; last row column 1
				subplot(rows, cols, 1 + cols * (rows - 1))
				title('$\chi^2$')
				plot_text('K.S. test p-value = %.2g %%\n$\mathrm{dof}=n-{\#}p = $%d\n$N\cdot(\\bar{\chi}^2 - \mathrm{dof}) = $%.2g $\sqrt{2\cdot\mathrm{dof}}$' % (100*pvalue, n-len(p0), chidist), loc=1)
				hist(chisq, **histkw)

				# histogram of execution time; last row column 2
				subplot(rows, cols, 2 + cols * (rows - 1))
				title('time [ms]')
				plot_text('Average time = %.2g ms' % (1000*times.mean()), loc=1)
				hist(times*1000, **histkw)

				# example data; last row column 3 (or first row last column)
				subplot(rows, cols, (3 + cols * (rows - 1)) if len(p0) >= 2 else cols)
				title('Example fit')
				fx = linspace(min(xmean), max(xmean), 1000)
				plot(fx, f(fx, *p0), '-', color='lightgray', linewidth=5, label='$y=%s$' % flatex, zorder=1)
				errorbar(x, y, dy, dx, fmt=',k', capsize=0, label='Data', zorder=2)
				plot(fx, f(fx, *par), 'r-', linewidth=1, label='Fit', zorder=3, alpha=1)
				# plot_text('$y=%s$\n$y=%s$' % (flatex, sp.latex(fsym(xsym, *p0))), fontsize=20)
				ticklabel_format(style='sci', axis='both', scilimits=(-3,3))
				legend(loc=0, fontsize='small')
				
				# save figure and show
				savefig(lab.nextfilename(gcf().canvas.get_window_title(), '.pdf', prepath='Figures'))
				show()

if showpsplot:
	figure('Function %s, fit with method “%s”, parameter biases' % (fstr, fitfun), figsize=(14,10)).set_tight_layout(True)
	clf()
	for i in range(len(p0s)): # p_i = fitted
		for j in range(len(p0s)): # p_j = true
			subplot(len(p0s), len(p0s), 1 + i * len(p0s) + j)
			K = [0] * len(p0s)
			K[j] = Ellipsis
			K = tuple(K)
			errorbar(p0s[j], fp[(0, 0) + K + (i,)] - (asarray(p0s[i]) if i == j else p0s[i][0]), sqrt(cp[(0, 0) + K + (i, i)]), fmt=',')
			xlabel('True $p_{%d}$' % j)
			ylabel('$p_{%d}\'-p_{%d}$' % (i, i))
			pstr = ''
			for k in range(len(p0s)):
				if k != j:
					pstr += '$p_{%d}$ = %.2g\n' % (k, p0s[k][0])
			plot_text(pstr)
			ticklabel_format(style='sci', axis='both', scilimits=(-3,3))
			grid()
				
	savefig(lab.nextfilename(gcf().canvas.get_window_title(), '.pdf', prepath='Figures'))
	show()
	# figure('Slope 3d, fit with method “%s”' % fitfun).set_tight_layout(True)
	# clf()
	# subplot(111, projection='3d')
	# X, Y = meshgrid(ms, qs)
	# gca().plot_surface(X, Y, (fp[:,:,0,0,0]-ms[:,newaxis]).T, rstride=1, cstride=1)
	# xlabel('True m')
	# ylabel('True q')
	# gca().set_zlabel('$m\'-m$')

if showpsdtplot:
	
	figure('Function %s, fit with method “%s”, parameters vs. errors' % (fstr, fitfun), figsize=(14,10)).set_tight_layout(True)
	clf()
	
	ds = [
		[dxs, dys, 'x', 'y', (Ellipsis, 0)],
		[dys, dxs, 'y', 'x', (0, Ellipsis)]
	]
	for i in range(len(p0s)):
		for j in range(2):
			subplot(len(p0s), 2, 2*i + j + 1)
			if i == 0:
				pstr = ''
				for k in range(len(p0s)):
					pstr += '$p_{%d}$ = %.2g\n' % (k, p0s[k][0])
				pstr += '$\sqrt{\sum\Delta %s^2/n}=$%.2g' % (ds[j][3], sqrt((ds[j][1][0]**2).sum() / n))
				plot_text(pstr)
			if i == len(p0s) - 1:
				xlabel('$\sqrt{\sum\Delta %s^2/n}$' % ds[j][2])
			if j == 0:
				ylabel('$p_{%d}\'-p_{%d}$' % (i, i))
			sel = ds[j][4] + tuple([0] * len(p0s)) + (i,)
			Y = fp[sel] - asarray(p0s[i])
			DY = sqrt(cp[sel + (i,)])
			errorbar(sqrt((ds[j][0]**2).sum(axis=-1) / n), Y, DY, fmt=',')
			pvalue = st.chi2.sf(sum((Y / DY)**2), len(Y))
			plot_text('p-value = %.2g %%' % (pvalue * 100), loc=1)
		
	savefig(lab.nextfilename(gcf().canvas.get_window_title(), '.pdf', prepath='Figures'))
	show()
