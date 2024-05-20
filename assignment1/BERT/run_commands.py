import subprocess

batch_sizes = [16, 32]
learning_rates = [2e-5, 3e-5, 5e-5]

for bs in batch_sizes:
    for lr in learning_rates:
        subprocess.run([
            'python', 'main.py',
            '--bs', str(bs),
            '--lr', str(lr)
        ])