import numpy as np
from scipy.integrate import nquad
import scipy.stats as stats
from pylab import *
import lab
import time
import scipy.linalg as la

# TODO/NOTE
#
# fit_bayes_1
# va esponenziale nel numero di parametri, è ragionevole fino a due parametri
# 1 parametro: un cazzo
# 2 parametri: 1 secondo
# 3 parametri: 2 minuti
# 4 parametri: non ci ho provato, 4 ore?
# l'np.array dentro fint non mi piace, e bisogna capire come scegliere epsrel / epsabs in modo più fino, queste cose richiedono di modificare nquad.
# integrare in coordinate sferiche?
# si può ancora guadagnare un fattore costante parallelizzando gli integrali, girando un po' le cose dovrebbero essere TUTTI indipendenti a meno di overflow
# bisogna poter dare i priori!
# di sicuro questa funzione non va bene con gli errori sulle x perché lì ogni x è un parametro
#
# bisogna fare un fit_bayes_2 che usi monte carlo
# metodi possibili:
# 0) sampling uniforme ignorante
# 1) sampling gaussiano basato su p0, covp0
# 2) sampling furbo che spara N uniformi, se la varianza è ancora alta divide lungo x_0 e va avanti ricorsivo (ciclando x_1, x_2, etc altrimenti la suddivisione a cubetti porta un andamento esponenziale)
# 3) impacchettare emcee (mi spaventa scegliere in automatico il burn-in)

f = lambda x, a, b: a * x**2 + b * x
p0 = (-1, -1)
x = linspace(0, 1, 1000)
dy = np.array([.05] * len(x))

y = f(x, *p0) + randn(len(x)) * dy

def format_par_cov(par, cov):
	pars = lab.xe(par, np.sqrt(np.diag(cov)))
	corr = lab.fit_norm_cov(cov) * 100
	corrwidth = 8
	s = ''
	for i in range(len(par)):
		for j in range(len(par)):
			width = max(corrwidth, len(pars[j])) + 1
			if i == j:
				sadd = pars[i]
			elif i < j:
				sadd = "%*.1f %%" % (corrwidth - 2, corr[i, j])
			else:
				sadd = ''
			s += ' ' * (width - len(sadd)) + sadd
			if j != len(par) - 1:
				s += ' '
			elif i != len(par) - 1:
				s += '\n'
	return s

def diagonalize_cov(cov):
	"""
	returns V, A with A diagonal such that
	V A V.T = cov
	
	other properties:
	V A^(-1) V.T = cov^(-1)
	p' = V.T p
	p'.T A^(-1) p' = p.T cov^(-1) p
	
	cov must be symmetric!
	
	"""
	w, V = la.eigh(cov)
	
	A = zeros((len(w), len(w)))
	np.fill_diagonal(A, w)
	
	return V, A

def fit_bayes_1(f, x, y, dy, p0, cov0):
	p0 = asarray(p0)
	cov0 = asarray(cov0)
	
	V, A = diagonalize_cov(cov0)
	dp0 = sqrt(diag(A))
	p0 = V.T.dot(p0)
		
	# 10^(-3/2) because we show errors with "1.5" digits
	# 1/2 to round correctly
	# 1/2 because the are two sources of error, one is sistematic
	# 1/√2 because everything gets multiplied by normalization
	relerr = 10 ** (-3/2) * 1/2 * 1/2 * 1/np.sqrt(2)
	
	def flim(epsrel):
		radius = abs(stats.norm.ppf(epsrel / 2)) + 1
		radius2 = radius ** 2
		def lim_circle(*p):
			x0 = np.sqrt(radius2 - np.sum(np.array(p)**2))
			return (-x0, x0)
		def lim_square(*p):
			return (-radius, radius)
		return [lim_circle] * len(p0)
	fopts = lambda epsrel: dict(epsabs=0, epsrel=epsrel)
	
	idy2 = 1 / dy ** 2
	chi20 = np.sum((y - f(x, *(V.dot(p0))))**2 * idy2)
	def L(*p):
		# must be centered around the maximum
		# also we must be careful with overflow/underflow
		# return np.exp(-0.5 * np.sum(((y - f(x, *p)) / dy)**2))
		# return np.prod(np.exp(-((y - f(x, *p)) / dy)**2 / 2) / (np.sqrt(2 * np.pi) * dy))
		# return np.exp((-np.sum(((y - f(x, *p)) / dy)**2) + len(x) - len(p0)) / 2)
		return np.exp((-np.sum((y - f(x, *(V.dot(p))))**2 * idy2) + chi20) / 2)
	
	# NORMALIZATION
	
	# we change variable such that the function is already with integral about 1 (we hope)
	# and that the horizontal scale is about 1
	def fint(*p):
		p = np.array(p) * dp0 + p0
		return L(*p)
	
	epsrel = 1 / np.sqrt(len(p0)) * relerr * min(dp0 / np.abs(p0)) # quadrature relative errors sum
	lim = flim(epsrel)
	
	start = time.time()
	N, dN, out = nquad(fint, lim, opts=fopts(epsrel), full_output=True)
	deltat = time.time() - start
	
	print('Normalization = %s, epsrel: %.2g, neval: %d' % (lab.xe(N, dN), epsrel, out['neval']))

	fa = linspace(-4, 4, 1000)
	
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
	
	# AVERAGE

	par = np.empty(len(p0))
	for i in range(len(par)):
		def fint(*p):
			p = np.array(p) * dp0 + p0
			return p[i] * L(*p)
		epsrel = 1 / np.sqrt(len(p0)) * relerr * dp0[i] / abs(p0[i])
		lim = flim(epsrel)
		start = time.time()
		par[i], err, out = nquad(fint, lim, opts=fopts(epsrel), full_output=True)
		par[i] /= N
		err /= N
		deltat = time.time() - start
		
		print('Average_%d = %s, epsrel: %.2g, neval: %d' % (i, lab.xe(par[i], err), epsrel, out['neval']))
		
		subplot(len(p0), len(p0) + 1, i * (len(p0) + 1) + 1)
		p = zeros(len(p0))
		def fplot(pi):
			p[i] = pi
			return fint(*p)
		plot(fa, [fplot(A) for A in fa], label='Time: %.3g s' % (deltat,))
		legend()
	
	# COVARIANCE

	cov = np.empty((len(par), len(par)))
	for i in range(len(par)):
		for j in range(i + 1):
			def fint(*p):
				p = np.array(p) * dp0 + par
				return (p[i] - par[i]) * (p[j] - par[j]) * L(*p)
			epsrel = 1 / np.sqrt(len(p0)) * relerr * np.sqrt(2)
			epsabs = 0
			lim = flim(epsrel)
			if i != j:
				epsabs = epsrel * dp0[i] * dp0[j]
				epsrel = 0
			start = time.time()
			cov[i, j], err, out = nquad(fint, lim, opts=dict(epsrel=epsrel, epsabs=epsabs), full_output=True)
			cov[i, j] /= N
			err /= N
			deltat = time.time() - start
			cov[j, i] = cov[i, j]
			
			print('Covariance_%d,%d = %s, epsrel: %.2g, epsabs: %.2g, neval: %d' % (i, j, lab.xe(cov[i, j], err), epsrel, epsabs, out['neval']))
		
			subplot(len(p0), len(p0) + 1, i * (len(p0) + 1) + j + 2)
			p = zeros(len(p0))
			def fplot(pij, sign):
				p[i] = pij
				p[j] = sign * pij
				return fint(*p)
			plot(fa, [fplot(A, -1) for A in fa], label='Time: %.3g s' % (deltat,))
			legend()
			if i != j:
				subplot(len(p0), len(p0) + 1, j * (len(p0) + 1) + i + 2)
				plot(fa, [fplot(A, 1) for A in fa], label='Time: %.3g s' % (deltat,))
				legend()
	
	par = V.dot(par)
	cov = V.dot(cov).dot(V.T)

	return par, cov

par, cov = lab.fit_generic(f, x, y, dy=dy, p0=p0)

print(format_par_cov(par, cov))

par, cov = fit_bayes_1(f, x, y, dy, par, cov)

print(format_par_cov(par, cov))

show()
