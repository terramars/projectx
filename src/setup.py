from distutils.core import setup
import py2exe

includes = ['numpy.fft']
excludes = ['scipy.linalg','scipy.optimize','scipy.interpolate','scipy.integrate','scipy.fftpack']
opts = {
		'py2exe':{
			'excludes' : excludes,
			'includes' : includes,
			'bundle_files' : 2} }

setup(windows = ['camera_feed.py'],
		options = opts)