import numpy as np
from scipy.integrate import nquad
import scipy.stats as stats
from pylab import *
import lab

f = lambda x, a, b: a * x + b
p0 = (-7, 5)
x = linspace(0, 1, 1000)
dy = [0.05] * len(x)

y = f(x, *p0) + randn(len(x)) * dy

def fit_bayes(f, x, y, dy, p0, dp0):
	p0 = asarray(p0)
	dp0 = asarray(dp0)
	
	prod_dp0 = np.prod(dp0)
	
	def L(*p):
		# must be centered around the maximum
		# return np.exp(-0.5 * np.sum(((y - f(x, *p)) / dy)**2))
		# return np.prod(np.exp(-((y - f(x, *p)) / dy)**2 / 2) / (np.sqrt(2 * np.pi) * dy))
		# return np.exp((-np.sum(((y - f(x, *p)) / dy)**2) + len(x) - len(p0)) / 2)
		return np.exp((-np.sum(((y - f(x, *p)) / dy)**2) + len(x)) / 2)

	def fint(*p):
		p = np.array(p) * dp0 + p0
		return L(*p)
	
	N = nquad(fint, [(-np.inf, np.inf)] * len(p0))[0] * prod_dp0
	print(N)

	fa = linspace(-3, 3, 1000)
	
	figure(0)
	clf()
	plot(fa, [fint(A) for A in fa])
	
	figure(1)
	clf()

	par = np.empty(len(p0))
	for i in range(len(par)):
		C = prod_dp0 / (N * p0[i])
		def fint(*p):
			p = np.array(p) * dp0 + p0
			return p[i] * L(*p) * C
		par[i] = nquad(fint, [(-np.inf, np.inf)] * len(p0))[0] * p0[i]
		
		subplot(len(p0), len(p0) + 1, i * (len(p0) + 1) + 1)
		plot(fa, [fint(A) for A in fa])

	cov = np.empty((len(par), len(par)))
	for i in range(len(par)):
		for j in range(i + 1):
			C = prod_dp0 / (N * dp0[i] * dp0[j])
			def fint(*p):
				p = np.array(p) * dp0 + par
				return (p[i] - par[i]) * (p[j] - par[j]) * L(*p) * C
			cov[i, j] = nquad(fint, [(-np.inf, np.inf)] * len(p0))[0] * dp0[i] * dp0[j]
			cov[j, i] = cov[i, j]
			
			subplot(len(p0), len(p0) + 1, i * (len(p0) + 1) + j + 2)
			plot(fa, [fint(A) for A in fa])

	return par, cov

par, cov = lab.fit_linear(x, y, dy=dy, offset=True)

print(lab.fit_norm_cov(cov))
print(lab.xe(par, sqrt(diag(cov))))	

par, cov = fit_bayes(f, x, y, dy, par, sqrt(diag(cov)))

print(lab.fit_norm_cov(cov))
print(lab.xe(par, sqrt(diag(cov))))	

show()
