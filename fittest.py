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
showplot = True # show plot after monte carlo run with fixed data parameters
showmqplot = False # show parameter biases with fixed data
showmqdxdyplot = False # show parameter vs. errors
stattest = False # perform statistical test after monte carlo
# ms = linspace(-2,2,20) # slope
# qs = linspace(-2,2,20) # offset
# ms = array([1])
# qs = array([1])
p0s = [
	[1],
	[1],
] # true parameters, axis 0 = parameter, axis 1 = values
f = lambda x, a, b: a * sp.exp(x / b) # sympy function
# f = lambda x, m, q : m * x + q
# f = lambda x, m : m * x
n = 100 # number of points
# mcns = linspace(sqrt(100), sqrt(1000), 40)**2 # monte carlo runs
mcn = 1000
fitfun = 'odr' # ev, odr, odrpack, curve_fit, hoch
xmean = linspace(0, 1, n)
dys = outer([1], ones(n)*.01)
dxs = outer([1], ones(n)*.01)
# dxs = outer(linspace(1, 10, len(mcns)), linspace(1,100,n)*.001)
# dx = st.norm.rvs(size=n)*.02+.1
# dy = st.norm.rvs(size=n)*.02+.1
# dy = 0
####################

psym = [sp.Symbol('p_%d' % i, real=True) for i in range(len(p0s))]
xsym = sp.Symbol('x', real=True)
syms = [xsym] + psym

# format function in LaTeX and 1D text
flatex = sp.latex(f(*syms))
subscr = '₀₁₂₃₄₅₆₇₈₉'
psubsym = [sp.Symbol('p%c' % subscr[i], real=True) for i in range(len(p0s))]
fstr = str(f(xsym, *psubsym)).replace('*', '·')

# compute derivatives and lambdify
dfdx = sp.lambdify(syms, f(*syms).diff(xsym), "numpy")
dfdps = [sp.lambdify(syms, f(*syms).diff(psym[i]), "numpy") for i in range(len(psym))]
dfdp_rt = empty((len(p0s), len(xmean)))
def dfdp(x, *p):
	for i in range(len(p)):
		dfdp_rt[i] = dfdps[i](x, *p)
	return dfdp_rt
dfdpdxs = [sp.lambdify(syms, f(*syms).diff(xsym).diff(psym[i]), "numpy") for i in range(len(psym))]
dfdpdx_rt = empty((len(p0s), len(xmean)))
def dfdpdx(x, *p):
	for i in range(len(p)):
		dfdpdx_rt[i] = dfdpdxs[i](x, *p)
	return dfdpdx_rt

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
					par, cov = lab.fit_generic_xyerr(f, dfdx, x, y, dx, dy, p0=p0)
				elif fitfun == 'odr':
					par, cov = lab.fit_generic_xyerr3(f, dfdx, dfdp, dfdpdx, x, y, dx, dy, p0)
				elif fitfun == 'hoch':
					par, cov = fit_generic_xyerr4(f, fi, dfdp, dfidp, x, y, dx, dy, p0)
				elif fitfun == 'odrpack':
					par, cov = lab.fit_generic_xyerr2(f, dfdx, dfdp, x, y, dx, dy, p0=p0)
				elif fitfun == 'curve_fit':
					par, cov = curve_fit(f, x, y, sigma=dy, p0=p0, absolute_sigma=True, jac=lambda *args: dfdp(*args).T)
				end = time.time()
				
				# save results
				if showplot or stattest:
					times[i] = end - start
					chisq[i] = ((y - f(x, *par))**2 / (dy**2 + (dfdx(x, *par)*dx)**2)).sum()
				pars[i] = par
				covs[i] = cov
			
			# save results
			pm = pars.mean(axis=0)
			pc = empty(2 * [len(p0)])
			for i in range(len(p0)):
				for j in range(len(p0)):
					pc[i, j] = ((pars[:,i] - pm[i]) * (pars[:,j] - pm[j])).sum() / len(pars)
			ps = sqrt(diag(pc))

			fp[(l, ll) + K] = pm
			cp[(l, ll) + K] = pc / sqrt(len(pars))
			
			if stattest or showplot:
				
				fs = array([sqrt(covs[:,i,i]).mean() for i in range(len(p0))])
				fc = empty(2 * [len(p0)])
				for i in range(len(p0)):
					for j in range(len(p0)):
						fc[i,j] = covs[:,i,j].mean()

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
				
				figure('Fit with method “%s”, function %s' % (fitfun, fstr), figsize=(14,10)).set_tight_layout(True)
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
						plot_text('True $\\rho =$ %.2g\nDistance = %.2g %%' % (pc[i,j] / ps[i] / ps[j], 100*(fc[i,j] - pc[i,j]) / pc[i,j]))
						hist(covs[:,i,j] / sqrt(covs[:,i,i]*covs[:,j,j]) - pc[i,j] / ps[i] / ps[j], bins=nbins, color=hcolor)
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
				fx = linspace(min(x), max(x), 1000)
				plot(fx, f(fx, *par), 'r-', label='$y=%s$' % flatex)
				# plot_text('$y=%s$\n$y=%s$' % (flatex, sp.latex(fsym(xsym, *p0))), fontsize=20)
				ticklabel_format(style='sci', axis='both', scilimits=(-3,3))
				legend(loc=0)
				
				# save figure and show
				filename = gcf().canvas.get_window_title().replace('/', '∕')
				i = 1
				while os.path.exists('%s-%02d.pdf' % (filename, i)):
					i += 1
				savefig('%s-%02d.pdf' % (filename, i))
				show()

if showmqplot:
	figure('Slope, fit with method “%s”' % fitfun).set_tight_layout(True)
	clf()
	subplot(211)
	errorbar(ms, fp[:, 0, 0, 0, 0] - ms, dp[:, 0, 0, 0, 0], fmt=',')
	xlabel('True m')
	ylabel('$m\'-m$, q=%g' % qs[0])
	grid()
	subplot(212)
	errorbar(qs, fp[0, :, 0, 0, 0] - ms[0], dp[0, :, 0, 0, 0], fmt=',')
	xlabel('True q')
	ylabel('$m\'-m$, m=%g' % ms[0])
	grid()
	tight_layout()
	figure('Slope 3d, fit with method “%s”' % fitfun).set_tight_layout(True)
	clf()
	subplot(111, projection='3d')
	X, Y = meshgrid(ms, qs)
	gca().plot_surface(X, Y, (fp[:,:,0,0,0]-ms[:,newaxis]).T, rstride=1, cstride=1)
	xlabel('True m')
	ylabel('True q')
	gca().set_zlabel('$m\'-m$')

	figure('Offset, fit with method “%s”' % fitfun).set_tight_layout(True)
	clf()
	subplot(211)
	errorbar(ms, fp[:, 0, 0, 0, 1] - qs[0], dp[:, 0, 0, 0, 1], fmt=',')
	xlabel('True m')
	ylabel('$q\'-q$, q=%g' % qs[0])
	grid()
	subplot(212)
	errorbar(qs, fp[0, :, 0, 0, 1] - qs, dp[0, :, 0, 0, 1], fmt=',')
	xlabel('True q')
	ylabel('$q\'-q$, m=%g' % ms[0])
	grid()
	tight_layout()
	figure('Offset 3d, fit with method “%s”' % fitfun).set_tight_layout(True)
	clf()
	subplot(111, projection='3d')
	gca().plot_surface(X, Y, (fp[:,:,0,0,1]-qs[newaxis,:]).T, rstride=1, cstride=1)
	xlabel('True m')
	ylabel('True q')
	gca().set_zlabel('$q\'-q$')
	show()

if showmqdxdyplot:	
	figure('Slope vs dx, %s' % fitfun).set_tight_layout(True)
	clf()
	subplot(211)
	title('True m = %g, q = %g, sum(dy^2) = %g' % (ms[0], qs[0], (dys[0]**2).sum()))
	xlabel('sum of dx**2')
	ylabel('Fitted m')
	errorbar((dxs**2).sum(axis=-1), fp[0,0,:,0,0], dp[0,0,:,0,0], fmt=',')
	subplot(212)
	title('True m = %g, q = %g, sum(dx^2) = %g' % (ms[0], qs[0], (dxs[0]**2).sum()))
	xlabel('sum of dy**2')
	ylabel('Fitted m')
	errorbar((dys**2).sum(axis=-1), fp[0,0,0,:,0], dp[0,0,0,:,0], fmt=',')


	figure('Offset vs dx, %s' % fitfun).set_tight_layout(True)
	clf()
	subplot(211)
	title('True m = %g, q = %g, sum(dy^2) = %g' % (ms[0], qs[0], (dys[0]**2).sum()))
	xlabel('sum of dx**2')
	ylabel('Fitted q')
	errorbar((dxs**2).sum(axis=-1), fp[0,0,:,0,1], dp[0,0,:,0,1], fmt=',')
	subplot(212)
	title('True m = %g, q = %g, sum(dx^2) = %g' % (ms[0], qs[0], (dxs[0]**2).sum()))
	xlabel('sum of dy**2')
	ylabel('Fitted q')
	errorbar((dys**2).sum(axis=-1), fp[0,0,0,:,1], dp[0,0,0,:,1], fmt=',')

	show()
