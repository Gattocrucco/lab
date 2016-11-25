import lab
import scipy.stats as st
from pylab import *
from scipy.optimize import curve_fit
import time
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D
from strangefit import *

#### PARAMETERS ####
showplot = True # show plot after monte carlo run with fixed data parameters
showmqplot = False # show parameter biases with fixed data
showmqdxdyplot = False # show parameter vs. errors
stattest = False # perform statistical test after monte carlo
# ms = linspace(-2,2,20) # slope
# qs = linspace(-2,2,20) # offset
ms = array([1])
qs = array([1])
n = 100 # number of points
# mcns = linspace(sqrt(100), sqrt(1000), 40)**2 # monte carlo runs
mcns = [1000]
fitfun = 'numhoch' # ev, odr, numodr, odrpack, x, y, hoch, numhoch
xmean = linspace(0, 1, n)
dys = outer([100], st.norm.rvs(size=n)*.0002+.001)
dxs = outer([100], st.norm.rvs(size=n)*.0002+.001)
# dxs = outer(linspace(1, 10, len(mcns)), linspace(1,100,n)*.001)
# dx = st.norm.rvs(size=n)*.02+.1
# dy = st.norm.rvs(size=n)*.02+.1
# dy = 0
####################

linfun = lambda x, m, q: m * x + q
syms = sp.symbols('x m q', real=True)
# dxlinfun = sp.lambdify(syms, linfun(*syms).diff(syms[0]), "numpy")
dxlinfun = lambda x, m, q: m * ones(len(x))
dplinfun = lambda x, m, q: array([x, ones(len(x))])
# dplinfun = sp.lambdify(syms, [linfun(*syms).diff(syms[1]), linfun(*syms).diff(syms[2])], "numpy")
dpxlinfun = lambda x, m, q: array([ones(len(x)), zeros(len(x))])
ilinfun = lambda y, m, q: y/m - q/m
def dpilinfun(y, m, q):
	rt = empty((2, len(y)))
	rt[0] = (q - y) / m**2 
	rt[1] = -1 / m
	return rt
# dpilinfun = lambda y, m, q: array([-y/m**2, [-1/m]*len(y)])
# def dpilinfun(y, m, q):
# 	rt = empty((2, len(y)))
# 	rt[0] = y
# 	rt[0] /= -m**2
# 	rt[1] = -1/m
# 	return rt
eta = lab.etastart()
fp, dp = empty((2, len(ms), len(qs), len(dxs), len(dys), 2))
cp = empty((len(ms), len(qs), len(dxs), len(dys)))
for ll in range(len(dys)):
	dy = dys[ll]
	for l in range(len(dxs)):
		mcn = int(mcns[l])
		dx = dxs[l]
		for k in range(len(qs)):
			q = qs[k]
			for j in range(len(ms)):
				m = ms[j]
				# generate mean data
				ymean = m * xmean + q
				# run fits
				chisq = empty(mcn)
				pars = empty((mcn, 2))
				covs = empty((mcn, 2, 2))
				times = empty(mcn)
				for i in range(mcn):
					lab.etastr(eta, (i + mcn * (j + len(ms) * (k + len(qs) * (l + len(dxs) * ll)))) / (mcn * len(ms) * len(qs) * len(dxs) * len(dys)), mininterval=5)
					# generate data
					deltax = st.norm.rvs(size=n)
					x = xmean + dx * deltax
					deltay = st.norm.rvs(size=n)
					y = ymean + dy * deltay
					# fit
					start = time.time()
					if fitfun == 'numodr':
						par, cov = lab.fit_generic_xyerr3(linfun, dxlinfun, dplinfun, dpxlinfun, x, y, dx, dy, (m, q))
					elif fitfun == 'ev':
						par, cov = lab.fit_linear(x, y, dx, dy, method='ev', conv_diff=1e-6, max_cycles=10)
					elif fitfun == 'odr':
						par, cov = lab._fit_affine_odr(x, y, dx, dy)
					elif fitfun == 'numhoch':
						par, cov = fit_generic_xyerr4(linfun, ilinfun, dplinfun, dpilinfun, x, y, dx, dy, (m, q))
					elif fitfun == 'odrpack':
						par, cov = lab.fit_generic_xyerr2(linfun, dxlinfun, dplinfun, x, y, dx, dy, (m, q))
					elif fitfun == 'hoch':
						par, cov = fit_linear_hoch(x, y, dx, dy)
					elif fitfun == 'x':
						par, cov = lab._fit_affine_xerr(x, y, dx)
					elif fitfun == 'y':
						par, cov = lab._fit_affine_yerr(x, y, dy)
					end = time.time()
					# save results
					if showplot or stattest:
						times[i] = end - start
						M, Q = par
						chisq[i] = ((y - (M * x + Q))**2 / (dy**2 + (M*dx)**2)).sum()
					pars[i] = par
					covs[i] = cov
				
				pm = pars.mean(axis=0)
				ps = pars.std(ddof=1, axis=0)
				pc = ((pars[:,0] - pm[0]) * (pars[:,1] - pm[1])).sum() / len(pars) 
	
				fp[j, k, l, ll] = pm
				dp[j, k, l, ll] = ps / sqrt(mcn)
				cp[j, k, l, ll] = pc
				
				if stattest or showplot:
				
					fs = array([sqrt(covs[:,0,0]).mean(), sqrt(covs[:,1,1]).mean()])
					fss = array([sqrt(covs[:,0,0]).std(ddof=1), sqrt(covs[:,1,1]).std(ddof=1)])
					fc = covs[:,0,1].mean()
					fcs = covs[:,0,1].std(ddof=1)

					pdist = (pm - array([m,q])) / ps
					sdist = (fs - ps) / fss
					cdist = (fc - pc) / fcs
					chidist = (chisq.mean() - (n-2)) / chisq.std(ddof=1)
					chisdist = (chisq.std(ddof=1) - sqrt(2*(n-2)))
					print(pdist, chidist, sdist, cdist)
					
					pvalues = [
						st.kstest(pars[:,0], 'norm', (m, fs[0]))[1],
						st.kstest(pars[:,1], 'norm', (q, fs[1]))[1],
						st.kstest(chisq, 'chi2', (n-2,))[1]
					]
					
					print(pvalues)
				
				def plot_text(string, loc=2):
					locs = [
						[],
						[.95, .95, 'right', 'top'],
						[.05, .95, 'left', 'top']
					]
					loc = locs[loc]
					text(loc[0], loc[1], string, horizontalalignment=loc[2], verticalalignment=loc[3], transform=gca().transAxes)
				
				if showplot:
					figure('Fit with method “%s”, number of points %d, number of runs %d, avg dx ∕ x width = %.2g %%' % (fitfun, n, mcn, 100 * abs(dx).mean() / (max(xmean) - min(xmean))), figsize=(14,10)).set_tight_layout(True)
					clf()
					nbins = int(sqrt(min(mcn, 1000)))
					hcolor = (.9,.9,.9)
					subplot(421)
					title('$m\'-m$')
					plot_text('True $m = $%g\nDistance = %.2g $\sigma$' % (m, pdist[0]))
					hist(pars[:, 0]-m, bins=nbins, color=hcolor)
					subplot(423)
					title('$q\'-q$')
					plot_text('True $q = $%g\nDistance = %.2g $\sigma$' % (q, pdist[1]))
					hist(pars[:, 1]-q, bins=nbins, color=hcolor)
					subplot(422)
					title('$(\Delta m\')\' - \Delta m\'$')
					plot_text('True $\sigma = $%.2g\nDistance = %.2g %%' % (ps[0], 100*(fs[0] - ps[0]) / ps[0]))
					hist(sqrt(covs[:,0,0]) - ps[0], bins=nbins, color=hcolor)
					ticklabel_format(style='sci',axis='x',scilimits=(-2,3))
					subplot(424)
					title('$(\Delta q\')\' - \Delta q\'$')
					plot_text('True $\sigma = $%.2g\nDistance = %.2g %%' % (ps[1], 100*(fs[1] - ps[1]) / ps[1]))
					hist(sqrt(covs[:,1,1]) - ps[1], bins=nbins, color=hcolor)
					ticklabel_format(style='sci',axis='x',scilimits=(-2,3))
					subplot(425)
					title('$(m\',q\')-(m,q)$')
					plot(pars[:, 0]-m, pars[:, 1]-q, '.k', markersize=2)
					grid()
					subplot(426)
					title('$\\rho_{m\'q\'}\'-\\rho_{m\'q\'}$')
					plot_text('True $\\rho =$ %.2g\nDistance = %.2g %%' % (pc / ps[0] / ps[1], 100*(fc - pc) / pc))
					hist(covs[:, 0, 1] / sqrt(covs[:, 0, 0]*covs[:, 1, 1]) - pc / ps[0] / ps[1], bins=nbins, color=hcolor)
					ticklabel_format(style='sci',axis='x',scilimits=(-2,3))
					subplot(427)
					title('$\chi^2$')
					plot_text('KSTest p-value = %.2g %%\nTrue dof = %d\nDistance = %.2g $\sigma$' % (100*pvalues[2], n-2, chidist), loc=1)
					hist(chisq, bins=nbins, color=hcolor)
					subplot(428)
					title('time [ms]')
					plot_text('Average time = %.2g ms' % (1000*times.mean()), loc=1)
					hist(times*1000, bins=nbins, color=hcolor)
					savefig(gcf().canvas.get_window_title() + '.pdf')
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
