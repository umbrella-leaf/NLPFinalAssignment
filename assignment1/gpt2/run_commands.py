import subprocess

batch_sizes = [32, 64]
learning_rates = [1e-4, 1e-5]
non_linearities = ['swish', 'gelu']
reduction_factors = [16, 8]

# Run training for each config
for bs in batch_sizes:
    for lr in learning_rates:
        for non_lin in non_linearities:
            for red_factor in reduction_factors:
                subprocess.run([
                    'python', 'main.py',
                    '--bs', str(bs),
                    '--lr', str(lr),
                    '--non_lin', non_lin,
                    '--red_factor', str(red_factor)
                ])