# Wrapper to fix cached file issue
import sys
sys.path.insert(0, '.')
# Read and exec the actual test
data = open('test_schur_complement.py', 'rb').read()
# Fix the truncation if present
if data.rstrip().endswith(b'self.asser'):
    data = data.rstrip()
    data = data[:data.rfind(b'self.asser')] + b"""self.assertGreater(len(sep), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
"""
exec(compile(data, 'test_schur_complement.py', 'exec'))
