import numpy as np
from scipy.integrate import nquad
import scipy.stats as stats
from pylab import *
import lab
import time

# TODO
# sembra funzionare (posto che gli si dia ottimi parametri iniziali, in pratica devo dargli già il risultato e lui mi fa la correzione bayesiana), però con 2 parametri è già lentissimo (dovrebbe essere esponenziale nel numero di parametri).

f = lambda x, a, b: a * x + b
p0 = (-7, 5)
x = linspace(0, 1, 10000)
dy = np.array([0.05] * len(x))

y = f(x, *p0) + randn(len(x)) * dy

def fit_bayes(f, x, y, dy, p0, dp0):
	p0 = asarray(p0)
	dp0 = asarray(dp0)
	
	prod_dp0 = np.prod(dp0)
	
	# 10^(-3/2) because we show errors with "1.5" digits
	# 1/2 to round correctly
	# 1/2 because the are two sources of error, one is sistematic
	# 1/√2 because everything gets multiplied by normalization
	relerr = 10 ** (-3/2) * 1/2 * 1/2 * 1/np.sqrt(2)
	
	idy2 = 1 / dy ** 2
	def L(*p):
		# must be centered around the maximum
		# also we must be careful with overflow/underflow
		# return np.exp(-0.5 * np.sum(((y - f(x, *p)) / dy)**2))
		# return np.prod(np.exp(-((y - f(x, *p)) / dy)**2 / 2) / (np.sqrt(2 * np.pi) * dy))
		# return np.exp((-np.sum(((y - f(x, *p)) / dy)**2) + len(x) - len(p0)) / 2)
		return np.exp((-np.sum((y - f(x, *p))**2 * idy2) + len(x)) / 2)
	
	# NORMALIZATION
	
	# we change variable such that the function is already with integral about 1 (we hope)
	# and that the horizontal scale is about 1
	def fint(*p):
		p = np.array(p) * dp0 + p0
		return L(*p) * prod_dp0
	
	epsrel = 1 / np.sqrt(len(p0)) * relerr * min(dp0 / np.abs(p0)) # quadrature relative errors sum
	# lim = 1 / sqrt(epsrel) # from cebichev inequality
	lim = abs(stats.norm.ppf(epsrel / 2)) # normal approximation
	
	start = time.time()
	N, dN = nquad(fint, [(-lim, lim)] * len(p0), opts=dict(epsabs=0, epsrel=epsrel))
	deltat = time.time() - start
	
	print('Normalization = %s, Limits: ±%.1f, relative error: %.2g' % (lab.xe(N, dN), lim, epsrel))

	fa = linspace(-3, 3, 1000)
	
	figure(0)
	clf()
	suptitle('Normalization')
	for i in range(len(p0)):
		subplot(len(p0), 1, i + 1)
		p = zeros(len(p0))
		def fplot(pi):
			p[i] = pi
			return fint(*p)
		plot(fa, [fplot(A) for A in fa], label='Time: %.3g s' % (deltat,))
		if i == 0:
			legend()
	
	figure(1)
	clf()
	suptitle('Average and covariance')

	par = np.empty(len(p0))
	for i in range(len(par)):
		C = prod_dp0 / (N * p0[i])
		def fint(*p):
			p = np.array(p) * dp0 + p0
			return p[i] * L(*p) * C
		epsrel = 1 / np.sqrt(len(p0)) * relerr * dp0[i] / abs(p0[i])
		lim = abs(stats.norm.ppf(epsrel / 2))
		start = time.time()
		par[i], err = np.array(nquad(fint, [(-lim, lim)] * len(p0), opts=dict(epsabs=0, epsrel=epsrel))) * p0[i]
		deltat = time.time() - start
		
		print('Average_%d = %s, Limits: ±%.1f, relative error: %.2g' % (i, lab.xe(par[i], err), lim, epsrel))
		
		subplot(len(p0), len(p0) + 1, i * (len(p0) + 1) + 1)
		p = zeros(len(p0))
		def fplot(pi):
			p[i] = pi
			return fint(*p)
		plot(fa, [fplot(A) for A in fa], label='Time: %.3g s' % (deltat,))
		legend()

	cov = np.empty((len(par), len(par)))
	for i in range(len(par)):
		for j in range(i + 1):
			C0 = dp0[i] * dp0[j]
			C = prod_dp0 / (N * C0)
			def fint(*p):
				p = np.array(p) * dp0 + par
				return (p[i] - par[i]) * (p[j] - par[j]) * L(*p) * C
			epsrel = 1 / np.sqrt(len(p0)) * relerr / sqrt(2)
			lim = abs(stats.norm.ppf(epsrel / 2))
			start = time.time()
			cov[i, j], err = array(nquad(fint, [(-lim, lim)] * len(p0), opts=dict(epsabs=0, epsrel=epsrel))) * C0
			deltat = time.time() - start
			cov[j, i] = cov[i, j]
			
			print('Covariance_%d,%d = %s, Limits: ±%.1f, relative error: %.2g' % (i, j, lab.xe(cov[i, j], err), lim, epsrel))
		
			subplot(len(p0), len(p0) + 1, i * (len(p0) + 1) + j + 2)
			p = zeros(len(p0))
			def fplot(pij):
				p[i] = pij
				p[j] = pij
				return fint(*p)
			plot(fa, [fplot(A) for A in fa], label='Time: %.3g s' % (deltat,))
			legend()

	return par, cov

par, cov = lab.fit_generic(f, x, y, dy=dy, p0=p0)

print(lab.fit_norm_cov(cov))
print(lab.xe(par, sqrt(diag(cov))))	

par, cov = fit_bayes(f, x, y, dy, par, sqrt(diag(cov)))

print(lab.fit_norm_cov(cov))
print(lab.xe(par, sqrt(diag(cov))))	

show()
