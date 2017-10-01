DESCRIPTION = """\
NeoAnalysis is a pure Python package for electrophysiological data processing and analysis
"""
maintainer = 'Bo Zhang & Ji Dai'
maintainer_email = 'zhangbo_1008@163.com & Ji Dai: dai_jijj@hotmail.com'

setupOpts = dict(
    name='NeoAnalysis',
    description='Electrophysiological data processing and analysis toolbox',
    long_description=DESCRIPTION,
    license='MIT',
    url = 'https://github.com/neoanalysis/NeoAnalysis',
    author=maintainer,
    author_email=maintainer_email,
    classifiers = [
        'License :: MIT license',
        'Programming Language :: Python :: 3.5',
    ],
)


import distutils.dir_util
from distutils.command import build
import os, sys, re
try:
    import setuptools
    from setuptools import setup
    from setuptools.command import install
except ImportError:
    from distutils.core import setup
    from distutils.command import install


# Work around mbcs bug in distutils.
# http://bugs.python.org/issue10945
import codecs
try:
    codecs.lookup('mbcs')
except LookupError:
    ascii = codecs.lookup('ascii')
    func = lambda name, enc=ascii: {True: enc}.get(name=='mbcs')
    codecs.register(func)


path = os.path.split(__file__)[0]
sys.path.insert(0, os.path.join(path, 'tools'))
import setupHelpers as helpers

## generate list of all sub-packages
allPackages = (helpers.listAllPackages(pkgroot='NeoAnalysis'))

## Decide what version string to use in the build
version, forcedVersion, gitVersion, initVersion = helpers.getVersionStrings(pkg='NeoAnalysis')
version = '1.0.0'


class Build(build.build):
    """
    * Clear build path before building
    """
    def run(self):
        global path

        ## Make sure build directory is clean
        buildPath = os.path.join(path, self.build_lib)
        if os.path.isdir(buildPath):
            distutils.dir_util.remove_tree(buildPath)
    
        ret = build.build.run(self)
        

class Install(install.install):
    """
    * Check for previously-installed version before installing
    * Set version string in __init__ after building. This helps to ensure that we
      know when an installation came from a non-release code base.
    """
    def run(self):
        global path, version, initVersion, forcedVersion, installVersion
        
        name = self.config_vars['dist_name']
        print(name)
        path = os.path.join(self.install_libbase, 'NeoAnalysis')
        if os.path.exists(path):
            raise Exception("It appears another version of %s is already "
                            "installed at %s; remove this before installing." 
                            % (name, path))
        print("Installing to %s" % path)
        rval = install.install.run(self)

        
        # If the version in __init__ is different from the automatically-generated
        # version string, then we will update __init__ in the install directory
        if initVersion == version:
            return rval
        
        try:
            initfile = os.path.join(path, '__init__.py')
            data = open(initfile, 'r').read()
            open(initfile, 'w').write(re.sub(r"__version__ = .*", "__version__ = '%s'" % version, data))
            installVersion = version
        except:
            sys.stderr.write("Warning: Error occurred while setting version string in build path. "
                             "Installation will use the original version string "
                             "%s instead.\n" % (initVersion)
                             )
            if forcedVersion:
                raise
            installVersion = initVersion
            sys.excepthook(*sys.exc_info())
    
        return rval


setup(
    version=version,
    cmdclass={'build': Build, 
              'install': Install,
              'deb': helpers.DebCommand, 
              'test': helpers.TestCommand,
              'debug': helpers.DebugCommand,
              'mergetest': helpers.MergeTestCommand,
              'style': helpers.StyleCommand},
    packages=allPackages,
    package_data={'':['SPC_Darwin_64bit','SPC_Linux_64bit','SPC_Linux_32bit','SPC_Windows_32bit.exe','SPC_Windows_64bit.exe']},
    install_requires = [
        'numpy>=1.11.3',
        'scipy>=0.18.1',
        'matplotlib>=2.0.0',
        'scikit-learn>=0.18.1',
        'quantities>=0.11.1',
        'pyopengl>=3.1.0',
        'pandas>=0.19.2',
        'h5py>=2.6.0',
        'statsmodels>=0.6.1',
        'seaborn>=0.7.1',
        'PyWavelets>=0.5.2'
    ],

    **setupOpts
)

