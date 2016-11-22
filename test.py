import lab
import scipy.stats as st
from pylab import *
from scipy.optimize import curve_fit
import time
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D

#### PARAMETERS ####
showplot = True
showmqplot = False
showmqdxdyplot = False
# ms = linspace(-2,2,40) # slope
# qs = linspace(-2,2,40) # offset
ms = array([1])
qs = array([1])
n = 1000 # number of points
# mcns = linspace(sqrt(100), sqrt(1000), 40)**2 # monte carlo runs
mcns = [1000]
fitfun = lab._fit_affine_odr
xmean = linspace(0, 1, n)
dys = outer([1], st.norm.rvs(size=n)*.0002+.001)
dxs = outer([1], linspace(1,1000,n)*.001)
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
					if fitfun == lab.fit_generic_xyerr3:
						par, cov = fitfun(linfun, dxlinfun, dplinfun, dpxlinfun, x, y, dx, dy, (m, q))
					elif fitfun == lab.fit_linear:
						par, cov = fitfun(x, y, dx, dy, conv_diff=1e-6, max_cycles=10)
					elif fitfun == lab.fit_generic_xyerr4:
						par, cov = fitfun(linfun, ilinfun, dplinfun, dpilinfun, x, y, dx, dy, (m, q))
					elif fitfun == lab.fit_generic_xyerr2:
						par, cov = fitfun(linfun, dxlinfun, dplinfun, x, y, dx, dy, (m, q))
					elif fitfun == lab.fit_linear_hoch:
						par, cov = fitfun(x, y, dx, dy)
					elif fitfun == lab._fit_affine_odr:
						par, cov = fitfun(x, y, dx, dy)
					end = time.time()
					# save results
					if showplot:
						times[i] = end - start
						M, Q = par
						chisq[i] = ((y - (M * x + Q))**2 / (dy**2 + (M*dx)**2)).sum()
					pars[i] = par
					covs[i] = cov
	
				fp[j, k, l, ll] = pars.mean(axis=0)
				dp[j, k, l, ll] = pars.std(ddof=1, axis=0) / sqrt(mcn)
		
				if showplot:
					figure('Fit with m=%g, q=%g, fun=%s' % (m, q, fitfun.__name__)).set_tight_layout(True)
					clf()
					subplot(421)
					title('$m\'-m$')
					hist(pars[:, 0]-m, bins=int(sqrt(mcn)))
					subplot(423)
					title('$q\'-q$')
					hist(pars[:, 1]-q, bins=int(sqrt(mcn)))
					subplot(422)
					title('$\Delta m$')
					hist(sqrt(covs[:, 0, 0]), bins=int(sqrt(mcn)))
					subplot(424)
					title('$\Delta q$')
					hist(sqrt(covs[:, 1, 1]), bins=int(sqrt(mcn)))
					subplot(425)
					title('$(m\',q\')-(m,q)$')
					plot(pars[:, 0]-m, pars[:, 1]-q, '.k', markersize=2)
					grid()
					subplot(426)
					title('$\\rho mq$')
					hist(covs[:, 0, 1] / sqrt(covs[:, 0, 0]*covs[:, 1, 1]), bins=int(sqrt(mcn)))
					subplot(427)
					title('$\chi^2$')
					hist(chisq, bins=int(sqrt(mcn)))
					subplot(428)
					title('time [ms]')
					hist(times*1000, bins=int(sqrt(mcn)))
					show()

if showmqplot:
	figure('Slope, fit with fun=%s' % fitfun.__name__)
	clf()
	subplot(211)
	errorbar(ms, fp[:, 0, 0, 0, 0], dp[:, 0, 0, 0, 0], fmt=',')
	xlabel('True m')
	ylabel('Fitted m, q=%g' % qs[0])
	grid()
	subplot(212)
	errorbar(qs, fp[0, :, 0, 0, 0], dp[0, :, 0, 0, 0], fmt=',')
	xlabel('True q')
	ylabel('Fitted m, m=%g' % ms[0])
	grid()
	tight_layout()
	figure('Slope 3d, fit with fun=%s' % fitfun.__name__)
	clf()
	subplot(111, projection='3d')
	X, Y = meshgrid(ms, qs)
	gca().plot_surface(X, Y, fp[:,:,0,0,0].T)
	xlabel('True m')
	ylabel('True q')
	gca().set_zlabel('Fitted m')

	figure('Offset, fit with fun=%s' % fitfun.__name__)
	clf()
	subplot(211)
	errorbar(ms, fp[:, 0, 0, 0, 1], dp[:, 0, 0, 0, 1], fmt=',')
	xlabel('True m')
	ylabel('Fitted q, q=%g' % qs[0])
	grid()
	subplot(212)
	errorbar(qs, fp[0, :, 0, 0, 1], dp[0, :, 0, 0, 1], fmt=',')
	xlabel('True q')
	ylabel('Fitted q, m=%g' % ms[0])
	grid()
	tight_layout()
	figure('Offset 3d, fit with fun=%s' % fitfun.__name__)
	clf()
	subplot(111, projection='3d')
	gca().plot_surface(X, Y, fp[:,:,0,0,1].T)
	xlabel('True m')
	ylabel('True q')
	gca().set_zlabel('Fitted q')
	show()

if showmqdxdyplot:	
	figure('Slope vs dx, %s' % fitfun.__name__).set_tight_layout(True)
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


	figure('Offset vs dx, %s' % fitfun.__name__).set_tight_layout(True)
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
