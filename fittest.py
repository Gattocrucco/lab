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

# TODO
# usare su tutti i subplot sigma, percentual e mean distance

#### PARAMETERS ####
showplot = True # show plot after monte carlo run with fixed parameters
showpsplot = False # show parameter biases with fixed data
showpsdtplot = False # show parameter vs. errors
stattest = False # perform statistical test after monte carlo
weightedaverage = True
p0s = [ # true parameters, axis 0 = parameter, axis 1 = values
	# linspace(-1,1,10),
	# logspace(0,1,10),
	[1],
	[1]
]
fs = [ # sympy functions
	lambda x, a, b: a * sp.exp(x / b),
	lambda x, m, q: m * x + q,
	lambda x, m: m * x
]
f = fs[1] # function to fit
mcn = 1000 # number of repetitions (monte carlo)
fitfun = 'odrpack' # ev, odr, odrpack, curve_fit, hoch
xmean = linspace(0, 3, 100) # true x
n = len(xmean) # number of points
dys = outer([1], ones(n)*.1) # errors, axis 0 = dataset, axis 1 = point
dxs = outer([10], ones(n)*.1)
####################

# initialize symbols
psym = [sp.Symbol('p_%d' % i, real=True) for i in range(len(p0s))]
xsym = sp.Symbol('x', real=True)
syms = [xsym] + psym

# format function in LaTeX and 1D text
flatex = sp.latex(f(*syms))
psubsym = [sp.Symbol('p%s' % lab.num2sub(i), real=True) for i in range(len(p0s))]
fstr = str(f(xsym, *psubsym)).replace('**', '^').replace('*', '·')

# compute derivatives and lambdify
# dfdx = sp.lambdify(syms, f(*syms).diff(xsym), "numpy")
# dfdps = [sp.lambdify(syms, f(*syms).diff(psym[i]), "numpy") for i in range(len(psym))]
# dfdp_rt = empty((len(p0s), len(xmean)))
# def dfdp(x, *p):
# 	for i in range(len(p)):
# 		dfdp_rt[i] = dfdps[i](x, *p)
# 	return dfdp_rt
# dfdpdxs = [sp.lambdify(syms, f(*syms).diff(xsym).diff(psym[i]), "numpy") for i in range(len(psym))]
# dfdpdx_rt = empty((len(p0s), len(xmean)))
# def dfdpdx(x, *p):
# 	for i in range(len(p)):
# 		dfdpdx_rt[i] = dfdpdxs[i](x, *p)
# 	return dfdpdx_rt

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
			if weightedaverage:
				icovs = empty(covs.shape)
				for i in range(len(icovs)):
					icovs[i] = np.linalg.inv(covs[i])
				pc = np.linalg.inv(icovs.sum(axis=0))
				wpar = empty(pars.shape)
				for i in range(len(wpar)):
					wpar[i] = icovs[i].dot(pars[i])
				pm = pc.dot(wpar.sum(axis=0))
				pc *= len(pars)
				ps = sqrt(diag(pc))
			else:
				pm = pars.mean(axis=0)
				pc = empty(2 * [len(p0)])
				for i in range(len(p0)):
					for j in range(len(p0)):
						pc[i, j] = ((pars[:,i] - pm[i]) * (pars[:,j] - pm[j])).sum() / len(pars)
				ps = sqrt(diag(pc))

			fp[(l, ll) + K] = pm
			cp[(l, ll) + K] = pc / len(pars)
			
			if stattest or showplot:
				
				fs = array([sqrt(covs[:,i,i]).mean() for i in range(len(p0))])
				fc = empty(2 * [len(p0)])
				for i in range(len(p0)):
					for j in range(len(p0)):
						fc[i,j] = covs[:,i,j].mean()
				frho = empty(2 * [len(p0)])
				for i in range(len(p0)):
					for j in range(len(p0)):
						frho[i,j] = (covs[:,i,j] / sqrt(covs[:,i,i] * covs[:,j,j])).mean()
				prho = empty(2 * [len(p0)])
				for i in range(len(p0)):
					for j in range(len(p0)):
						prho[i,j] = pc[i,j] / sqrt(pc[i,i] * pc[j,j])

				pdist = (pm - array(p0)) / ps
				chidist = (chisq.mean() - (n-len(p0))) / chisq.std(ddof=1)
				print(pdist, chidist)
				
				pvalue = st.kstest(chisq, 'chi2', (n-len(p0),))[1]
				
				print(pvalue)
			
			def plot_text(string, loc=2, **kw):
				locs = [
					[],
					[.95, .95, 'right', 'top'],
					[.05, .95, 'left', 'top']
				]
				loc = locs[loc]
				text(loc[0], loc[1], string, horizontalalignment=loc[2], verticalalignment=loc[3], transform=gca().transAxes, **kw)
			
			if showplot:
				
				figure('Function %s, fit with method “%s”' % (fstr, fitfun), figsize=(14,10)).set_tight_layout(True)
				clf()
				nbins = int(sqrt(min(mcn, 1000)))
				maxscatter = 1000
				hcolor = (.9,.9,.9)
				rows = max(1 + len(p0), 3)
				cols = 1 + len(p0)
				
				# histogram of parameter; left column
				for i in range(len(p0)):
					subplot(rows, cols, 1 + i * cols)
					title('$p_%d\'-p_%d$' % (i, i))
					plot_text('True $p_%d = $%g\nDistance = %.2g $\sigma$' % (i, p0[i], pdist[i]))
					hist(pars[:,i] - p0[i], bins=nbins, color=hcolor)
					ticklabel_format(style='sci', axis='x', scilimits=(-3,3))
				
				# histogram of sigma; diagonal
				for i in range(len(p0)):
					subplot(rows, cols, 2 + i * (1 + cols))
					title('$\sigma_%d\' - \sigma_%d$' % (i, i))
					plot_text('True $\sigma = $%.2g\nDistance = %.2g %%' % (ps[i], 100*(fs[i] - ps[i]) / ps[i]))
					hist(sqrt(covs[:,i,i]) - ps[i], bins=nbins, color=hcolor)
					ticklabel_format(style='sci', axis='x', scilimits=(-3,3))
				
				# scatter plot of pairs of parameters; upper triangle
				for i in range(len(p0)):
					for j in range(i + 1, len(p0)):
						subplot(rows, cols, 2 + i * cols + j)
						title('$(p_%d\',p_%d\')-(p_%d,p_%d)$' % (i, j, i, j))
						X = pars[:, i] - p0[i]
						Y = pars[:, j] - p0[j]
						if len(X) > maxscatter:
							X = X[::int(ceil(len(X) / maxscatter))]
							Y = Y[::int(ceil(len(Y) / maxscatter))]
						plot(X, Y, '.k', markersize=2)
						grid()
						ticklabel_format(style='sci', axis='both', scilimits=(-3,3))
				
				# histogram of correlation; lower triangle
				for i in range(len(p0)):
					for j in range(i):
						subplot(rows, cols, 2 + i * cols + j)
						title('$\\rho_{%d%d}\'-\\rho_{%d%d}$' % (i, j, i, j))
						plot_text('True $\\rho =$ %.2g\nDistance = %.2g %%' % (prho[i,j], 100*(frho[i,j] - prho[i,j]) / prho[i,j]))
						hist(covs[:,i,j] / sqrt(covs[:,i,i]*covs[:,j,j]) - prho[i,j], bins=nbins, color=hcolor)
						ticklabel_format(style='sci', axis='x', scilimits=(-3,3))
				
				# histogram of chisquare; last row column 1
				subplot(rows, cols, len(p0) * cols + 1)
				title('$\chi^2$')
				plot_text('KSTest p-value = %.2g %%\nTrue dof = %d\nDistance = %.2g $\sigma$' % (100*pvalue, n-len(p0), chidist), loc=1)
				hist(chisq, bins=nbins, color=hcolor)
				
				# histogram of execution time; last row column 2
				subplot(rows, cols, len(p0) * cols + 2)
				title('time [ms]')
				plot_text('Average time = %.2g ms' % (1000*times.mean()), loc=1)
				hist(times*1000, bins=nbins, color=hcolor)
				
				# example data; last row last column
				subplot(rows, cols, rows * cols)
				title('Example fit')
				errorbar(x, y, dy, dx, fmt=',k', capsize=0, label='Data')
				fx = linspace(min(xmean), max(xmean), 1000)
				plot(fx, f(fx, *p0), 'r-', label='$y=%s$' % flatex)
				# plot_text('$y=%s$\n$y=%s$' % (flatex, sp.latex(fsym(xsym, *p0))), fontsize=20)
				ticklabel_format(style='sci', axis='both', scilimits=(-3,3))
				legend(loc=0)
				
				# save figure and show
				savefig(lab.nextfilename(gcf().canvas.get_window_title(), '.pdf'))
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
				
	savefig(lab.nextfilename(gcf().canvas.get_window_title(), '.pdf'))
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
		
	savefig(lab.nextfilename(gcf().canvas.get_window_title(), '.pdf'))
	show()
