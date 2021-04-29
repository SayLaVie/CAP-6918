class Logger:
    def __init__(self, log_file, stdout=False):
        self.log_file = log_file
        self.stdout = stdout

    def write(self, line='\n'):
        with open(self.log_file, 'a') as f:
            f.write(f'{line}\n')
        if self.stdout:
            print(f'{line}')
