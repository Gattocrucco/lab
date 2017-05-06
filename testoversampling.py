from pylab import *
import lab
from scipy import stats

sigma = 50
delta = .5

S = 1/2

dist = stats.norm(scale=S)

figure('test oversampling')
clf()

xs = linspace(-sigma, sigma, 1000)

def _logerf(x, delta):
	if x > 0:
		return log(dist.sf(x - delta) - dist.sf(x + delta))
	else:
		return log(dist.cdf(x + delta) - dist.cdf(x - delta))

def logpdf(x, delta):
	xd = abs(x) - delta
	return dist.logpdf(0.5 * (sign(xd) + 1) * xd * sign(x))

logerf = vectorize(_logerf, otypes=[float], excluded=[1])

y0s = logerf(xs, delta)
y1s = dist.logpdf(xs)
y2s = logpdf(xs, delta * 0.8)

subplot(211)

plot(xs, y0s, label='logerf')
plot(xs, y1s, label='logpdf')
plot(xs, y2s, label='logpdf($x - \\delta$)')

legend(loc=0)

subplot(212)

plot(xs, y1s - y0s, label='logpdf $-$ logerf')
plot(xs, y2s - y0s, label='logpdf($x - \\delta$) $-$ logerf')
legend(loc=0)

show()
