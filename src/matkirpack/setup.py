from setuptools import setup
setup(name='matpack',
            version='0.95',
            description='packages for master thesis',
            url='http://github.com/matkir',
            author='Mathias Kirkerod',
            author_email='mathias.kirkerod@gmail.com',
            license='MIT',
            packages=['encdec','plotload','selector','masker','cutter','prepros'],
            zip_safe=False)
