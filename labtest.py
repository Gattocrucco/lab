import unittest
import lab

class TestFormatting(unittest.TestCase):

	def test_num2si(self):
		# generic check
		self.assertEqual(lab.num2si(1312), '1.312 k')
		# check crash on 0
		self.assertEqual(lab.num2si(0), '0')
		# check that format options are respected
		self.assertEqual(lab.num2si(1, format='%+g'), '+1')
	
	def test_util_format(self):
		# check that big-exponent values use exponential notation
		self.assertEqual(lab.util_format(1.23456789e-8, 1.1111e-10, pm=None, percent=False), '1.235(11)e-8')
		# check that number of digits is chosen correctly in case of uncertainty rounding
		self.assertEqual(lab.util_format(10, 0.99, pm=None, percent=False), '10.0(10)')
	
	def test_util_mm_esr(self):
		# check that big numbers are checked
		with self.assertRaises(ValueError):
			lab.util_mm_esr(1e8, unit='volt', metertype='kdm700')

	# def test_isupper(self):
	# 	self.assertTrue('FOO'.isupper())
	# 	self.assertFalse('Foo'.isupper())
	#
	# def test_split(self):
	# 	s = 'hello world'
	# 	self.assertEqual(s.split(), ['hello', 'world'])
	# 	# check that s.split fails when the separator is not a string
	# 	with self.assertRaises(TypeError):
	# 		s.split(2)

if __name__ == '__main__':
	unittest.main()
