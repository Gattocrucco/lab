import unittest
import lab
import numpy as np

class TestLab(unittest.TestCase):

    def test_num2si(self):
        # generic check
        self.assertEqual(lab.num2si(1312), '1.312 k')
        # check crash on 0
        self.assertEqual(lab.num2si(0), '0 ')
        # check that format options are respected
        self.assertEqual(lab.num2si(1, format='%+g'), '+1 ')
        # check that default rounding is sufficient
        self.assertEqual(lab.num2si(0.7), '700 m')
    
    def test_util_format(self):
        # check that big-exponent values use exponential notation
        self.assertEqual(lab.util_format(1.23456789e-8, 1.1111e-10, pm=None, percent=False), '1.235(11)e-8')
        # check that number of digits is chosen correctly in case of uncertainty rounding
        self.assertEqual(lab.util_format(10, 0.99, pm=None, percent=False), '10.0(1.0)')
        # check that percentual error is not negative
        self.assertEqual(lab.util_format(-1, 1, pm=None, percent=True), '-1.0(1.0) (100 %)')
        # check that percentual error is suppressed if mantissa is 0 when we are using compact error notation
        self.assertEqual(lab.util_format(0.001, 1, pm=None, percent=True), '0(10)e-1')
        # check that if mantissa is 0 and compact notation is not used the error has correct exponent
        self.assertEqual(lab.util_format(0, 1, pm='+-'), '(0 +- 10)e-1')
        self.assertEqual(lab.util_format(0, 10, pm='+-'), '0 +- 10')
        self.assertEqual(lab.util_format(0, 1e5, pm='+-'), '(0 +- 10)e+4')
    
    def test_util_mm_esr(self):
        # check that big numbers are checked
        with self.assertRaises(ValueError):
            lab.util_mm_esr(1e8, unit='volt', metertype='kdm700')
        # check that internal resistance of voltmeter is retrieved properly (error if fails)
        lab.util_mm_esr(1, unit='volt', metertype='kdm700')
    
    def test_fit_norm_cov(self):
        # just check that it works because there's always someone willing to rewrite this stupid function
        cov = [[4, -3], [-3, 16]]
        normalized_cov = [[1, -0.375], [-0.375, 1]]
        self.assertTrue(np.array_equal(lab.fit_norm_cov(cov), normalized_cov))

class TestFitCurve(unittest.TestCase):
    
    def test_line_defaults(self):
        # Perform the fit of a straight line with fit_linear and with
        # all the methods of fit_curve, with default options. Check
        # that results are identical.
        
        # Config parameters
        x = np.linspace(0, 1, 10)
        m = 2
        q = 3
        dy = 0.1 * np.ones(x.shape) # uniform errors because of leastsq
        methods = [
            'wleastsq',
            'leastsq',
            'odrpack',
            'linodr',
            'ev'
            # omit pymc3 because result is different
            # omit ml because it requires errors on x
        ]
        
        # Reference fit
        y = m * x + q + dy * np.random.randn(*x.shape)
        par, cov = lab.fit_linear(x, y, dy=dy)
        chisq = np.sum((y - (par[0] * x + par[1])) ** 2 / dy ** 2)
        
        # fits
        for method in methods:
            out = lab.fit_curve(lambda x, m, q: m * x + q, x, y, dy=dy, p0=[1, 1], method=method)
            assertions = []
            assertions.append(np.allclose(par, out.par))
            if method == 'leastsq':
                ratio = out.cov / cov
                assertions.append(np.allclose(0, ratio - np.mean(ratio)))
            else:
                assertions.append(np.allclose(cov, out.cov))
            assertions.append(np.allclose(chisq, out.chisq, atol=0.1))
            assertion = all(assertions)
            if not assertion:
                print('Fit different from reference fit.')
                print('Reference result:')
                print('chisq = {:.2f}'.format(chisq))
                print(lab.format_par_cov(par, cov))
                print('Method `{}` result:'.format(method))
                print('chisq = {:.2f}'.format(out.chisq))
                print(lab.format_par_cov(out.par, out.cov))
            self.assertTrue(assertion)

if __name__ == '__main__':
    unittest.main()
