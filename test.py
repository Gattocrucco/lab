import lab
import scipy.stats as st
from pylab import *

#### PARAMETERS ####
showplot = True
m = 1 # slope
q = 1 # offset
n = 1000 # number of points
mcn = 1000 # monte carlo runs
fitfun = lab.fit_linear
xmean = linspace(0, 1, n)
dx = array([.1] * n)
dy = array([.1] * n)
# dx = st.norm.rvs(size=n)*.1+.1
# dy = st.norm.rvs(size=n)*.05+.1
# dy = 0
####################

# generate mean data
ymean = m * xmean + q
# run fits and collect chisquares
chisq = empty(mcn)
pars = empty((mcn, 2))
covs = empty((mcn, 2, 2))
linfun = lambda x, m, q: m * x + q
dxlinfun = lambda x, m, q: m
dplinfun = lambda x, m, q: array([x, ones(len(x))])
dpxlinfun = lambda x, m, q: array([ones(len(x)), zeros(len(x))])
for i in range(mcn):
	# generate data
	deltax = st.norm.rvs(size=n)
	x = xmean + dx * deltax
	deltay = st.norm.rvs(size=n)
	y = ymean + dy * deltay
	# fit
	if fitfun == lab.fit_generic_xyerr3:
		par, cov = fitfun(linfun, dxlinfun, dplinfun, dpxlinfun, x, y, dx, dy, (1, 1))
	elif fitfun == lab.fit_linear:
		par, cov = fitfun(x, y, dx, dy)
	# save results
	M, Q = par
	chisq[i] = ((y - (M * x + Q))**2 / (dy**2 + (M*dx)**2)).sum()
	pars[i] = par
	covs[i] = cov

if showplot:
	figure(1)
	clf()
	subplot(421)
	title('$m$')
	hist(pars[:, 0], bins=int(sqrt(mcn)))
	subplot(423)
	title('$q$')
	hist(pars[:, 1], bins=int(sqrt(mcn)))
	subplot(422)
	title('$\Delta m$')
	hist(sqrt(covs[:, 0, 0]), bins=int(sqrt(mcn)))
	subplot(424)
	title('$\Delta q$')
	hist(sqrt(covs[:, 1, 1]), bins=int(sqrt(mcn)))
	subplot(425)
	title('$m,q$')
	plot(pars[:, 0], pars[:, 1], '.k', markersize=2)
	grid()
	subplot(426)
	title('$\Delta mq$')
	hist(covs[:, 0, 1] / sqrt(covs[:, 0, 0]*covs[:, 1, 1]), bins=int(sqrt(mcn)))
	subplot(414)
	title('$\chi^2$')
	hist(chisq, bins=int(sqrt(mcn)))
	tight_layout()
	show()
